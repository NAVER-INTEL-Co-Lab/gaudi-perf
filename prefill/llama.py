import os
from pprint import pprint
from statistics import mean, stdev

import torch
import torch.distributed as dist
import deepspeed
import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.distributed.hccl  # noqa
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import get_habana_frameworks_version
from habana_frameworks.torch.hpu import Event
from transformers import AutoConfig, AutoModelForCausalLM


adapt_transformers_to_gaudi()


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


def measure(
        model,
        full_name: str,
        seq_len: int,
        batch_size: int,
        num_steps: int,
) -> dict:
    config = AutoConfig.from_pretrained(full_name, torch_dtype=torch.bfloat16)

    flops = approx_llama_forward_macs(
        num_decoder_blocks=config.num_hidden_layers,
        sequence_length=seq_len,
        vocabulary_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        gated_ffn_act=True,
    )

    flops *= 2 * batch_size  # 1 MAC is approximately 2 FLOPs.
    device = torch.device("hpu")  # HPUs do not have numbers, unlike NVIDIA GPUs.
    x = torch.zeros(size=(batch_size, seq_len), dtype=torch.int64, device=device)

    forward_kwargs = dict(
        use_flash_attention=True,
        flash_attention_recompute=True,
        flash_attention_causal_mask=True,
        flash_attention_fast_softmax=True,
        attn_softmax_bf16=True,
    )

    for _ in range(16):  # Warmup
        model(x, **forward_kwargs)

    tics = [Event(enable_timing=True) for _ in range(num_steps)]
    tocs = [Event(enable_timing=True) for _ in range(num_steps)]

    for i in range(num_steps):
        tics[i].wait()  # Prevent out-of-order execution.
        tics[i].record()
        model(x, **forward_kwargs)
        tocs[i].record()
    ht.synchronize()

    mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
    tfps = [flops * 1e-9 / ms for ms in mss]
    tkps = [batch_size * seq_len * 1_000 / ms for ms in mss]

    info = {
        "Model Name": full_name,
        "Batch Size": batch_size,
        "Input Sequence Length": seq_len,
        "Average Latency (milliseconds)": mean(mss),
        "Synapse AI Version": get_habana_frameworks_version(),
        "PyTorch Version": torch.__version__,
        "DeepSpeed Version": deepspeed.__version__,
        "Mean Tokens per Second": mean(tkps),
        "Model Mean TFLOPS": mean(tfps),
        "Model Min TFLOPS": min(tfps),
        "Model Max TFLOPS": max(tfps),
        "Model STDEV TFLOPS": stdev(tfps),
    }
    return info


@torch.inference_mode()
def main(model_name: str, seq_len: int = 4096, num_steps: int = 32):
    deepspeed.init_distributed(dist_backend="hccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    dsd = "hpu" if world_size == 1 else "meta"
    with deepspeed.OnDevice(dtype=torch.bfloat16, device=dsd):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.eval()

    hpu_graph = world_size == 1
    model = deepspeed.init_inference(
        model,
        dtype=torch.bfloat16,
        tensor_parallel={"tp_size": world_size},
        enable_cuda_graph=hpu_graph,
        set_empty_params=True,  # Initialize empty parameters from meta-tensors.
        checkpoint=None,
    )

    info = measure(
        model=model,
        full_name=model_name,
        batch_size=1,
        seq_len=seq_len,
        num_steps=num_steps,
    )
    info.update({
        "Local Rank": local_rank, 
        "World Size (TP Degree)": world_size, 
        "Use HPU Graph": hpu_graph
    })
    dist.destroy_process_group()
    pprint(info)  # Only show the results from the main process.
