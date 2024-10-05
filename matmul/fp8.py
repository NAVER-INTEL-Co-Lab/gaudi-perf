"""
Available scaling methods can be found in the following link.
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html#supported-json-config-file-options

export PT_HPU_WEIGHT_SHARING=0
export LOG_LEVEL_HQT=1
export PT_HPU_LAZY_MODE=1

Run `python -m fire perf/matmul_bf16.py measure` to run for the provided shapes.
Run `python -m fire perf/matmul_bf16.py prof_matmul $m $k $n` for arbitrary shapes.
"""
from statistics import mean, stdev

import torch
from torch import nn
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
from neural_compressor.torch.quantization import (
    FP8Config, 
    convert, 
    prepare, 
    finalize_calibration,
)


htcore.hpu_set_env()


@torch.inference_mode()
def prof_matmul(
        m: int,
        k: int,
        n: int,
        num_steps: int = 64,
        fp8_config: str = "E4M3",
        scale_method: str = "maxabs_hw",
        measure_mode: bool = True,
):
    config_measure = FP8Config.from_dict({
        "fp8_config": fp8_config,
        "mode": "MEASURE",
        "observer": "maxabs",
        "allowlist": {"types": [], "names":  []},
        "blocklist": {"types": [], "names":  []},
        "dump_stats_path": f"./inc_output/{fp8_config}_{scale_method}_{m}_{k}_{n}",
    })

    config_quantize = FP8Config.from_dict({
        "fp8_config": fp8_config,
        "mode": "QUANTIZE",
        "observer": "maxabs",
        "scale_method": scale_method,
        "allowlist": {"types": [], "names":  []},
        "blocklist": {"types": [], "names":  []},
        "dump_stats_path": f"./inc_output/{fp8_config}_{scale_method}_{m}_{k}_{n}"
    })

    print("Initializing.")
    a = torch.randn(size=(m, k), device="hpu")
    # The Sequential layer is necessary because INC cannot take a single module.
    model = nn.Sequential(nn.Linear(in_features=k, out_features=n, bias=False, device="hpu"))

    if measure_mode:
        model_measure = prepare(model, config_measure)
        htcore.hpu_inference_initialize(model_measure, mark_only_scales_as_const=True)
        print("Starting measurement.")

        for _ in range(16):
            model_measure(a)

        finalize_calibration(model_measure)
        print("Finishing measurement.")
    else:
        model_quant = convert(model, config_quantize)
        htcore.hpu_inference_initialize(model_quant, mark_only_scales_as_const=True)
        model_quant = ht.hpu.wrap_in_hpu_graph(model_quant)

        print("Starting quantized run.")

        for _ in range(16):  # Warmup
            model_quant(a)
        
        tics = [ht.hpu.Event(enable_timing=True) for _ in range(num_steps)]
        tocs = [ht.hpu.Event(enable_timing=True) for _ in range(num_steps)]
        for i in range(num_steps):
            tics[i].wait()  # Prevent asynchronous launches.
            tics[i].record()
            model_quant(a)
            tocs[i].record()
        ht.hpu.synchronize()

        mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
        # Using the exact matmul flops equation.
        tflops = [1e-9 * m * n * (2 * k - 1) / ms for ms in mss]

        print(
            f"({m:5} x {k:5})x({k:5} x {n:5}) {fp8_config} {scale_method}: "
            f"Mean {mean(tflops):6.1f} TFLOPS, "
            f"Min {min(tflops):6.1f} TFLOPS, "
            f"Max {max(tflops):6.1f} TFLOPS, "
            f"STDEV {stdev(tflops):6.1f} TFLOPS, "
            f"MFU: {mean(tflops)/865*100:4.1f}%"
        )


def measure(num_steps: int = 64, fp8_config: str = "E4M3", scale_method: str = "maxabs_hw") -> None:
    mkn = (
        (16384, 8192, 1280),
        (16384, 1024, 8192),
        (16384, 8192, 7168),
        (16384, 3584, 8192),
        (2 ** 12, 2 ** 12, 2 ** 12),
        (2 ** 13, 2 ** 13, 2 ** 13),
        (2 ** 14, 2 ** 14, 2 ** 14),
        (2 ** 15, 2 ** 15, 2 ** 15),
    )
    kwargs = dict(num_steps=num_steps, fp8_config=fp8_config, scale_method=scale_method)
    for m, k, n in mkn:
        prof_matmul(m, k, n, measure_mode=True, **kwargs)
        prof_matmul(m, k, n, measure_mode=False **kwargs)
