"""
Note that accumulation always occurs in FP32 in the Gaudi, unlike NVIDIA GPUs.

Run `python -m fire matmul/bf16.py measure` to run for the provided shapes.
Run `python -m fire matmul/bf16.py prof_matmul $m $k $n` for user-provided shapes.
"""
from statistics import mean, median, stdev

import torch
from torch import nn, Tensor
import habana_frameworks.torch.hpu as ht


class MM(nn.Module):
    """
    This class profiles multiple GEMMs because for small GEMMs,
    the host overhead is greater than the computation overhead,
    resulting in inaccurate measurements that appear too slow.
    """
    def __init__(self, m: int, k: int, n: int, device, dtype, repeats: int = 1):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self.m1s = [torch.randn(size=(m, k), **dd) for _ in range(repeats)]
        self.m2s = [torch.randn(size=(n, k), **dd) for _ in range(repeats)]

    def forward(self):  # Equivalent to einsum("bmk,bnk->mn")
        outs = list()
        for m1, m2 in zip(self.m1s, self.m2s, strict=True):
            outs.append(torch.mm(input=m1, mat2=m2.T))
        return outs


@torch.inference_mode()
def prof_matmul(
        m: int,
        k: int,
        n: int,
        warmup_steps: int = 32,
        num_steps: int = 256,
        repeats: int = 1,  # Increase this value for small matrices below 4Kx4K.
        data_type: str = "bf16",
) -> None:
    if data_type == "fp32":
        dtype = torch.float32
    elif data_type == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid data type {data_type}.")

    mm = ht.wrap_in_hpu_graph(MM(m=m, k=k, n=n, device="hpu", dtype=dtype, repeats=repeats))
    for _ in range(warmup_steps):  # Warmup
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
    tfps = [1e-9 * repeats * m * n * (2 * k - 1) / ms for ms in mss]
    print(
        f"({m:5} x {k:5})x({k:5}x{n:5}): "
        f"Mean {mean(tfps):5.1f} TFLOPS, "
        f"Median {median(tfps):5.1f} TFLOPS, "
        f"Min {min(tfps):5.1f} TFLOPS, "
        f"Max {max(tfps):5.1f} TFLOPS, "
        f"STDEV {stdev(tfps):4.1f} TFLOPS, "
        f"Gaudi 2 MFU: {mean(tfps)/432*100:4.1f}%, "
        f"Gaudi 3 MFU: {mean(tfps)/1835*100:4.1f}%"
    )


def measure(num_steps: int = 256, data_type: str = "bf16") -> None:
    mknr = (
        (16384, 8192, 1280, 1),
        (16384, 1024, 8192, 1),
        (16384, 8192, 7168, 1),
        (16384, 3584, 8192, 1),
        (2 ** 12, 2 ** 12, 2 ** 12, 1),
        (2 ** 13, 2 ** 13, 2 ** 13, 1),
        (2 ** 14, 2 ** 14, 2 ** 14, 1),
    )
    for m, k, n, r in mknr:
        prof_matmul(m, k, n, num_steps=num_steps, data_type=data_type, repeats=r)
