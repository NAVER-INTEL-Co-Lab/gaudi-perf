"""
Run `python -m fire matmul/bf16.py measure` to run for the provided shapes.
Run `python -m fire matmul/bf16.py prof_matmul $m $k $n` for user provided shapes.
"""
from statistics import mean, stdev

import torch
from torch import nn, Tensor
import habana_frameworks.torch.hpu as ht


@torch.inference_mode()
def prof_matmul(
        m: int,
        k: int,
        n: int,
        num_steps: int = 64,
        data_type: str = "bf16",
) -> None:
    if data_type == "fp32":
        dtype = torch.float32
    elif data_type == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid data type {data_type}.")

    a = torch.randn(size=(m, k), device="hpu", dtype=dtype)
    w = torch.randn(size=(n, k), device="hpu", dtype=dtype)

    class MM(nn.Module):
        def __init__(self, mat1: Tensor, mat2: Tensor):
            super().__init__()
            self.m1 = mat1
            self.m2 = mat2

        def forward(self):
            return torch.mm(input=self.m1, mat2=self.m2)

    mm = ht.wrap_in_hpu_graph(MM(mat1=a, mat2=w.T))
    for _ in range(16):  # Warmup
        mm()

    tics = [ht.Event(enable_timing=True) for _ in range(num_steps)]
    tocs = [ht.Event(enable_timing=True) for _ in range(num_steps)]

    for i in range(num_steps):
        tics[i].wait()  # Wait until everything in front has finished.
        tics[i].record()
        mm()
        tocs[i].record()

    ht.synchronize()  # Synchronize after finishing.
    mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
    # Using the exact TFLOPS equation.
    tfps = [1e-9 * m * n * (2 * k - 1) / ms for ms in mss]
    print(
        f"({m:5} x {k:5})x({k:5}x{n:5}): "
        f"Mean {mean(tfps):5.1f} TFLOPS, "
        f"Min {min(tfps):5.1f} TFLOPS, "
        f"Max {max(tfps):5.1f} TFLOPS, "
        f"STDEV {stdev(tfps):7.3f} TFLOPS, "
        f"Gaudi v2 MFU: {mean(tfps)/432*100:4.1f}%, "
        f"Gaudi v3 MFU: {mean(tfps)/1835*100:4.1f}%"
    )


def measure(num_steps: int = 64, data_type: str = "bf16") -> None:
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
    for m, k, n in mkn:
        prof_matmul(m, k, n, num_steps=num_steps, data_type=data_type)
