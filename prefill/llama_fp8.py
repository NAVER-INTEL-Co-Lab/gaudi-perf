"""
Login to HuggingFace with `huggingface-cli login` if a gated repo is to be used.

Example run command below.

```bash
export PT_HPU_WEIGHT_SHARING=0
export LOG_LEVEL_HQT=1
export PT_HPU_LAZY_MODE=1

# First run with `measure_mode` enabled to get quantization statistics.
deepspeed --no_local_rank --num_gpus 8 \
    --module fire prefill/llama_fp8.py main \
    --model_name meta-llama/Llama-3.1-70B \
    --seq_len $((8 * 1024)) \
    --num_steps 32 \
    --measure_mode True

# Then run with `measure_mode` disabled for the actual run.
deepspeed --no_local_rank --num_gpus 8 \
    --module fire prefill/llama_fp8.py main \
    --model_name meta-llama/Llama-3.1-70B \
    --seq_len $((8 * 1024)) \
    --num_steps 32 \
    --measure_mode False
```

For power measurements, use one of the following commands on the host.
The container will likely not have `ipmitool` available.

```bash
sudo ipmitool dcmi power reading 5_sec
sudo ipmitool sensor get Total_Power
```
"""
import os
from pprint import pprint
from statistics import mean, stdev

import torch
import torch.distributed as dist
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.distributed.hccl  # noqa
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import get_habana_frameworks_version

import deepspeed
from habana_frameworks.torch.hpu import Event  # noqa
from transformers import AutoConfig, AutoModelForCausalLM


htcore.hpu_set_env()
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
        model_name: str,
        seq_len: int,
        batch_size: int,
        num_steps: int,
) -> dict:
    config = AutoConfig.from_pretrained(model_name)

    macs = approx_llama_forward_macs(
        num_decoder_blocks=config.num_hidden_layers,
        sequence_length=seq_len,
        vocabulary_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        gated_ffn_act=True,
    )

    flops = macs * 2 * batch_size  # 1 MAC is approximately 2 FLOPs.
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
        "Model Name": model_name,
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
        "Forward MAC Count": macs,
    }
    return info


@torch.inference_mode()
def main(
        model_name: str,
        num_steps: int = 16,
        batch_size: int = 1,
        seq_len: int = 4096,
        fp8_config: str = "E4M3",
        scale_method: str = "maxabs_hw",
        scale_format: str = "const",
        measure_mode: bool = True,
):
    deepspeed.init_distributed(dist_backend="hccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = torch.device("hpu")  # HPUs do not have numbers, unlike NVIDIA GPUs.
    dsd = device if world_size == 1 else "meta"
    with deepspeed.OnDevice(dtype=torch.bfloat16, device=dsd):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.eval()

    # Hack to prevent OOM on 16K sequence length for 34B.
    use_hpu_graph = (world_size == 1) and (seq_len < 16384)

    model = deepspeed.init_inference(
        model,
        dtype=torch.bfloat16,
        tensor_parallel={"tp_size": world_size},
        enable_cuda_graph=use_hpu_graph,
        max_out_tokens=1,
        min_out_tokens=1,
    )

    # Imports of these modules must come after DeepSpeed setup.
    from neural_compressor.torch.quantization import (
        FP8Config,
        convert,
        prepare,
        finalize_calibration,
    )
    dump_stats_path = f"./inc_output/measure_{model_name}_{seq_len}_{fp8_config}"
    blocklist_names = ["lm_head", "fused_scaled_dot_product_attention"]
    config_measure = FP8Config.from_dict({
        "fp8_config": fp8_config,
        "mode": "MEASURE",
        "observer": "maxabs",
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": blocklist_names},
        "dump_stats_path": dump_stats_path,
    })

    config_quantize = FP8Config.from_dict({
        "fp8_config": fp8_config,
        "mode": "QUANTIZE",
        "observer": "maxabs",
        "scale_method": scale_method,
        "scale_format": scale_format,
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": blocklist_names},
        "dump_stats_path": dump_stats_path,
    })

    if measure_mode:
        model_measure = prepare(model.module, config_measure)
        htcore.hpu_inference_initialize(model_measure, mark_only_scales_as_const=True)

        for _ in range(8):
            x = torch.randint(
                low=0,
                high=config.vocab_size,
                size=(batch_size, seq_len),
                dtype=torch.int64,
                device=device,
            )
            model_measure(x)
        finalize_calibration(model_measure)

    else:
        model_quant = convert(model.module, config_quantize)
        htcore.hpu_inference_initialize(model_quant, mark_only_scales_as_const=True)

        info = measure(
            model=model_quant,
            model_name=model_name,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
        )
        info.update({
            "Local Rank": local_rank,
            "World Size (TP Degree)": world_size,
            "Use HPU Graph": use_hpu_graph,
            "Binary Type": fp8_config,
            "Scale Method": scale_method,
            "Scale Format": scale_format,
        })
        dist.destroy_process_group()
        if local_rank == 0:  # Only show the results from the main process.
            pprint(info)
            print(f"\n\nMean TFLOPS/HPU: {info['Model Mean TFLOPS'] / world_size:.1f} TFLOPS\n\n")
