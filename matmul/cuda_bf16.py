"""
Run `python -m fire matmul/cuda_bf16.py measure` to run for the provided shapes.
Run `python -m fire matmul/cuda_bf16.py prof_matmul $m $k $n --repeats $r`
for user-provided shapes.
"""
from statistics import mean, median, stdev

import torch
from torch import nn


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

    @torch.compile(fullgraph=True, dynamic=False)
    def forward(self):
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

    mm = MM(m=m, k=k, n=n, device="cuda", dtype=dtype, repeats=repeats)

    for _ in range(warmup_steps):  # Warmup
        mm()

    tics = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    tocs = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    for i in range(num_steps):
        tics[i].wait()  # Prevent asynchronous launches.
        tics[i].record()
        mm()
        tocs[i].record()
    torch.cuda.synchronize()

    mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
    # Using the exact matmul flops equation.
    tfps = [1e-9 * repeats * m * n * (2 * k - 1) / ms for ms in mss]
    print(
        f"({m:5} x {k:5})x({k:5}x{n:5}): "
        f"Mean {mean(tfps):5.1f} TFLOPS, "
        f"Median {median(tfps):5.1f} TFLOPS, "
        f"Min {min(tfps):5.1f} TFLOPS, "
        f"Max {max(tfps):5.1f} TFLOPS, "
        f"STDEV {stdev(tfps):4.1f} TFLOPS, "
        f"A100 MFU: {mean(tfps) / 312 * 100:5.1f}%, "
        f"H100 MFU: {mean(tfps) / 989.4 * 100:4.1f}%"
    )


def measure(
        warmup_steps: int = 32,
        num_steps: int = 256,
        reduced_precision_reduction: bool = True,
):
    torch._dynamo.reset()  # Clear compilation cache.
    torch._dynamo.config.cache_size_limit = 64
    # Reduced precision reduction for BF16 GEMM is enabled by default in PyTorch.
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision_reduction

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
        prof_matmul(
            m=m,
            k=k,
            n=n,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repeats=r,
            data_type="bf16",
        )
