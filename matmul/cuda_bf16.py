"""
Run `python -m fire matmul/cuda_bf16.py measure` to run for the provided shapes.
"""
from statistics import mean, stdev

import torch


@torch.inference_mode()
def measure(num_steps: int = 256):
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
    mm = torch.compile(torch.matmul, fullgraph=True, dynamic=False)

    for m, k, n in mkns:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

        for _ in range(16):  # Warmup
            mm(x, y.T)

        tics = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        tocs = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        for i in range(num_steps):
            tics[i].wait()  # Prevent asynchronous launches.
            tics[i].record()
            mm(x, y.T)
            tocs[i].record()
        torch.cuda.synchronize()

        mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
        # Using the exact matmul flops equation.
        tfps = [1e-9 * m * n * (2 * k - 1) / ms for ms in mss]
        print(
            f"({m:5} x {k:5})x({k:5}x{n:5}): "
            f"Mean {mean(tfps):5.1f} TFLOPS, "
            f"Min {min(tfps):5.1f} TFLOPS, "
            f"Max {max(tfps):5.1f} TFLOPS, "
            f"STDEV {stdev(tfps):4.1f} TFLOPS, "
            f"A100 MFU: {mean(tfps)/312*100:4.1f}%, "
            f"H100 MFU: {mean(tfps)/989.4*100:4.1f}%"
        )
