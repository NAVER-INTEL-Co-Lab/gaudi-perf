"""
Login to HuggingFace with `huggingface-cli login` if a gated repo is to be used.

Single node run:
deepspeed --no_local_rank --num_gpus 8 \
    --module fire train/cuda_llama.py train \
    --model_name meta-llama/Llama-3.1-8B \
    --seq_len $((8 * 1024)) \
    --zero_stage 3 \
    --use_liger_kernel True
"""
import os
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F  # noqa
from torch.cuda import Event
import transformers
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
)
import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
import deepspeed


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=True,  # Fused QKV is different from HuggingFace.
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rope_emb = te_rope(max_seq_len=config.max_position_embeddings)

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=self.rope_emb,
                # Selective activation checkpointing is always enabled.
                checkpoint_core_attention=True,
            ),
        )


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Modified from the version in Transformer Engine for better performance.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        llama_for_causal_lm.lm_head = te.pytorch.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        llama_for_causal_lm.model.norm = te.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        llama_for_causal_lm.post_init()
        return llama_for_causal_lm


def approx_llama_forward_macs(
        num_decoder_blocks: int,
        sequence_length: int,
        vocabulary_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        exclude_causal_mask: bool = False,
        gated_ffn_act: bool = True,
        head_dim: int | None = None,
) -> int:
    assert hidden_size % num_attention_heads == 0
    assert num_attention_heads % num_key_value_heads == 0
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    # Query, Key, Value linear projection with Group Query Attention.
    num_qkv_heads = num_attention_heads + 2 * num_key_value_heads
    qkv_macs = sequence_length * hidden_size * head_dim * num_qkv_heads
    # Matrix multiply QK^T to get the self-attention matrix.
    qkt_macs = sequence_length * (head_dim * num_attention_heads) * sequence_length
    # Self-attention with the value tensor.
    sav_macs = sequence_length * sequence_length * (head_dim * num_attention_heads)
    # Post-attention projection with the output tensor.
    pap_macs = sequence_length * (head_dim * num_attention_heads) * hidden_size
    # Total number of MACs in attention.
    attn_macs = qkv_macs + qkt_macs + sav_macs + pap_macs
    # Exclude causal attention mask MACs from the total MAC count if desired.
    mask_shape = (sequence_length * (sequence_length - 1)) // 2
    mask_macs = head_dim * num_attention_heads * mask_shape
    attn_macs -= int(exclude_causal_mask) * 2 * mask_macs
    ffn_macs = sequence_length * hidden_size * intermediate_size
    # SwiGLU and other gated FFNs have another matrix multiply.
    ffn_macs *= 2 + int(gated_ffn_act)
    # Final matrix multiply over the vocabulary head,
    # which is sometimes tied to the input embedding weights.
    head_macs = sequence_length * hidden_size * vocabulary_size
    macs = head_macs + num_decoder_blocks * (attn_macs + ffn_macs)
    return macs  # MAC count.


def get_optim_groups(model: nn.Module, weight_decay: float = 0.1):
    wd = []
    nd = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            wd.append(p)
        elif p.ndim <= 1:
            nd.append(p)
        else:
            raise ValueError(f"Invalid shape {p.shape}.")

    return [
        {"params": wd, "weight_decay": weight_decay},
        {"params": nd, "weight_decay": 0.0},
    ]


def train(
        model_name: str,
        max_iters: int = 4,
        seq_len: int = 4096,
        zero_stage: int = 3,
        checkpoint_gradients: bool = False,
        batch_size: int = 64,
        micro_batch_size: int = 1,
        offload_optimizer: bool = True,
        offload_param: bool = True,
        full_bf16: bool = False,
        use_liger_kernel: bool = False,
        exclude_causal_mask: bool = False,
):
    deepspeed.init_distributed(dist_backend="nccl")
    if use_liger_kernel:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,  # For 8K only, frankly.
            rms_norm=False,
            swiglu=False,
            model=None,
        )

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    weight_decay = 0.1
    gradient_accumulation_steps = batch_size // world_size
    grad_accum_dtype = "bf16" if full_bf16 else "fp32"
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.,  # Setting to 0 should disable clipping.
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
        },
        "optimizer": {
            "type": "Adam",  # Uses AdamW by default.
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.95],
            },
        },
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype
        },
    }

    if offload_optimizer:
        ds_config.update({"offload_optimizer": {"device": "cpu", "pin_memory": True}})
    if offload_param:
        ds_config.update({"offload_param": {"device": "cpu", "pin_memory": True}})

    config = AutoConfig.from_pretrained(model_name)
    macs = approx_llama_forward_macs(
        num_decoder_blocks=config.num_hidden_layers,
        sequence_length=seq_len,
        vocabulary_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        exclude_causal_mask=exclude_causal_mask,
        gated_ffn_act=True,
    )
    device = torch.device(f"cuda:{local_rank}")

    with torch.cuda.device(device=device):
        config.use_cache = False  # Prevent errors from HF.
        model = TELlamaForCausalLM(config=config)
    model.train()
    if full_bf16:  # This is usually a terrible idea for training stability.
        model = model.to(torch.bfloat16)

    # Activate naive gradient checkpointing for HuggingFace models.
    # The model has selective activation checkpointing enabled even if
    # full block-level activation checkpointing is disabled.
    if checkpoint_gradients:
        model.gradient_checkpointing_enable()

    model_parameters = get_optim_groups(model, weight_decay=weight_decay)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,  # noqa
        config=ds_config,
    )
    engine.train()

    iter_num = 0
    x = torch.randint(config.vocab_size, size=(micro_batch_size, seq_len), device=device)
    y = torch.randint(config.vocab_size, size=(micro_batch_size, seq_len), device=device)
    tic = Event(enable_timing=True)
    toc = Event(enable_timing=True)

    if local_rank == 0:
        print("Starting training. The first few iterations may be slow due to warmup.")

    tic.record()
    while iter_num < max_iters:
        engine.backward(engine(x, labels=y).loss)
        engine.step()

        if engine.is_gradient_accumulation_boundary():
            iter_num += 1
            toc.record()
            toc.synchronize()
            # Average time per step in milliseconds.
            ms = tic.elapsed_time(toc)
            # Measuring model FLOPS instead of hardware FLOPS.
            tfps = 6 * macs * batch_size // world_size / ms * 1e-9
            if iter_num > 1:  # First step is warmup.
                if local_rank == 0:
                    print(
                        f"{model_name}, "
                        f"ZeRO-{zero_stage}, "
                        f"Sequence: {seq_len}, "
                        f"Batch: {batch_size}, "
                        f"Micro Batch: {micro_batch_size}",
                    )
                    print(f"Throughput: {tfps:.2f} TFLOPS")
                    print(f"Latency: {ms:.2f} milliseconds")
            tic.record()
