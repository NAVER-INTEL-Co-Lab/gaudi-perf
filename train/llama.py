"""
Note that lazy mode causes delays for long sequences at the beginning.

Login to HuggingFace with `huggingface-cli login` if a gated repo is to be used.

Single node run:
deepspeed --no_local_rank --num_gpus 8 \
    --module fire train/llama.py train \
    --model_name meta-llama/Llama-3.1-8B \
    --seq_len $((8 * 1024)) \
    --zero_stage 3 \
    --use_act_ckpt False \
    --offload_optimizer True \
    --offload_param True \
    --batch_size 64
"""
import os

import torch
from torch import nn
import habana_frameworks.torch as ht
from transformers import AutoConfig, AutoModelForCausalLM
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import deepspeed

adapt_transformers_to_gaudi()
deepspeed.init_distributed(dist_backend="hccl")


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
    qkv_macs = sequence_length * hidden_size * head_dim * (num_attention_heads + 2 * num_key_value_heads)
    # Matrix multiply QK^T to get the self-attention matrix.
    qkt_macs = sequence_length * (head_dim * num_attention_heads) * sequence_length
    # Self-attention with the value tensor.
    sav_macs = sequence_length * sequence_length * (head_dim * num_attention_heads)
    # Post-attention projection with the output tensor.
    pap_macs = sequence_length * (head_dim * num_attention_heads) * hidden_size
    # Total number of MACs in attention.
    attn_macs = qkv_macs + qkt_macs + sav_macs + pap_macs
    # Exclude causal attention mask MACs from the total MAC count if desired.
    causal_macs = (head_dim * num_attention_heads) * sequence_length * (sequence_length - 1) // 2
    attn_macs -= int(exclude_causal_mask) * 2 * causal_macs
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
        zero_stage: int,
        micro_batch_size: int = 1,
        num_steps: int = 4,
        seq_len: int = 4096,
        use_act_ckpt: bool = False,  # Full activation checkpointing.
        batch_size: int = 512,
        offload_optimizer: bool = True,
        offload_param: bool = True,
        full_bf16: bool = False,
) -> None:
    assert zero_stage in (1, 2, 3), zero_stage
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    gradient_accumulation_steps = batch_size // world_size
    grad_accum_dtype = "bf16" if full_bf16 else "fp32"
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.95],
            },
        },
        "data_types": {"grad_accum_dtype": grad_accum_dtype},
    }

    if offload_optimizer:
        ds_config.update({"offload_optimizer": {"device": "cpu", "pin_memory": True}})
    if offload_param:
        ds_config.update({"offload_param": {"device": "cpu", "pin_memory": True}})

    # The best configurations for Gaudi are different from those for NVIDIA GPUs.
    if zero_stage == 1:
        ds_config.update({"zero_optimization": {
            "stage": 1,
            "contiguous_gradients": False,
        }})
    elif 1 < zero_stage <= 3:
        ds_config.update({"zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": False,
            "reduce_scatter": False,
            "contiguous_gradients": True,
        }})
    else:
        raise ValueError(f"Invalid {zero_stage=}")

    device = torch.device("hpu")
    config = AutoConfig.from_pretrained(model_name)
    hs = config.hidden_size
    vs = config.vocab_size
    nl = config.num_hidden_layers
    it = config.intermediate_size
    nh = config.num_attention_heads
    kv = config.num_key_value_heads

    macs = approx_llama_forward_macs(
        num_decoder_blocks=nl,
        sequence_length=seq_len,
        vocabulary_size=vs,
        hidden_size=hs,
        intermediate_size=it,
        num_attention_heads=nh,
        num_key_value_heads=kv,
        gated_ffn_act=True,
    )

    # Options only available in Optimum Habana.
    config.fused_qkv = True
    config.flash_attention_fp8 = False  # Risky to use in practice.

    model_dtype = torch.bfloat16 if full_bf16 else torch.float32
    with deepspeed.OnDevice(dtype=model_dtype, device=device):
        config.use_cache = False  # Prevent errors from HF.
        model = AutoModelForCausalLM.from_config(config)
    model.train()  # Train mode during training.

    if use_act_ckpt:
        model.gradient_checkpointing_enable()

    engine, opt, *_ = deepspeed.initialize(
            model=model,
            model_parameters=get_optim_groups(model),  # noqa
            config=ds_config,
    )
    engine.train()

    fwd_kwargs = dict(
        use_flash_attention=True,
        flash_attention_recompute=True,  # Selective activation checkpointing?
        flash_attention_causal_mask=True,
        flash_attention_fast_softmax=True,
        attn_softmax_bf16=True,
    )

    iter_num = 0
    x = torch.randint(vs, size=(micro_batch_size, seq_len), device=device)
    y = torch.randint(vs, size=(micro_batch_size, seq_len), device=device)
    tic = ht.hpu.Event(enable_timing=True)
    toc = ht.hpu.Event(enable_timing=True)

    tic.record()
    while iter_num < num_steps:
        engine.backward(engine(x, labels=y, **fwd_kwargs).loss)
        engine.step()

        if engine.is_gradient_accumulation_boundary():
            iter_num += 1
            toc.record()
            toc.synchronize()
            # Average time per step in milliseconds.
            ms = tic.elapsed_time(toc)
            # 1 MAC is approx. 2 FLOPs and backward is double forward compute.
            # This leaves out checkpointing FLOPs as per the definition of MFU.
            tfps = 6 * macs * batch_size // world_size / ms * 1e-9
            if iter_num > 1:  # First step is not logged because it is warmup.
                if local_rank == 0:
                    print(
                        f"{model_name}, "
                        f"ZeRO-{zero_stage}, "
                        f"Sequence: {seq_len}, "
                        f"Checkpoint: {use_act_ckpt}, "
                        f"Batch: {batch_size}, "
                        f"Micro Batch: {micro_batch_size}",
                    )
                    print(f"Throughput: {tfps:.2f} TFLOPS")
                    print(f"Latency: {ms:.2f} milliseconds")
            tic.record()
