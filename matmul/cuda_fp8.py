"""
This code uses the PyTorch `_scaled_mm` function, which is currently under development.
Because of this, we recommend using the NGC PyTorch 24.09 image to execute this code.
Use `docker pull nvcr.io/nvidia/pytorch:24.09-py3` to fetch the 24.09 image.
If the exact version of PyTorch is not the same, the function may not work as expected.

Run the code below using the following command.

`python -m fire matmul/cuda_fp8.py measure`
"""
from statistics import mean, stdev

import torch


@torch.inference_mode()
def measure(
        num_steps: int = 64,
        fp8_config: str = "E4M3",  # Only E4M3 works for now.
        per_channel: bool = False, 
        use_fast_accum: bool = True, 
) -> None:
    torch._dynamo.reset()  # Clear compilation cache.
    torch._dynamo.config.cache_size_limit = 64
    mkns = (
        (16384, 8192, 1280),
        (16384, 1024, 8192),
        (16384, 8192, 7168),
        (16384, 3584, 8192),
        (2 ** 12, 2 ** 12, 2 ** 12),
        (2 ** 13, 2 ** 13, 2 ** 13),
        (2 ** 14, 2 ** 14, 2 ** 14),
        (2 ** 15, 2 ** 15, 2 ** 15),
    )
    smm = torch.compile(torch._scaled_mm, fullgraph=True, dynamic=False)

    fp8_dict = dict(E4M3=torch.float8_e4m3fn, E5M2=torch.float8_e5m2)
    device = torch.device("cuda")

    for m, k, n in mkns:
        x = torch.randn(m, k, device=device).to(dtype=fp8_dict.get(fp8_config))
        y = torch.randn(n, k, device=device).to(dtype=fp8_dict.get(fp8_config))
        if per_channel:
            s1 = torch.ones(m, 1, device=device)
            s2 = torch.ones(1, n, device=device)
        else:
            s1 = s2 = torch.ones(1, device=device)

        for _ in range(16):  # Warmup
            smm(x, y.T, s1, s2, out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum)

        tics = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        tocs = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        for i in range(num_steps):
            tics[i].wait()  # Prevent asynchronous launches.
            tics[i].record()
            smm(x, y.T, s1, s2, out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum)
            tocs[i].record()
        torch.cuda.synchronize()

        mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]

        # Using the exact matmul flops equation.
        tfps = [1e-9 * m * n * (2 * k - 1) / ms for ms in mss]
        print(
            f"({m:5} x {k:5})x({k:5} x {n:5}) E4M3 {use_fast_accum=}: "
            f"Mean {mean(tfps):6.1f} TFLOPS, "
            f"Min {min(tfps):6.1f} TFLOPS, "
            f"Max {max(tfps):6.1f} TFLOPS, "
            f"STDEV {stdev(tfps):4.1f} TFLOPS, "
            f"H100 MFU: {mean(tfps)/1978.9*100:4.1f}%"
        )
