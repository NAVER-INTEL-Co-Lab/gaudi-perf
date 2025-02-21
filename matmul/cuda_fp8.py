"""
This code uses the PyTorch `_scaled_mm` function, which is currently under development.
Because of this, we recommend using the NGC PyTorch 24.09 image to execute this code.
Use `docker pull nvcr.io/nvidia/pytorch:24.09-py3` to fetch the 24.09 image.
If the exact version of PyTorch is not the same, the function may not work as expected.

Run the code below using the following commands.

`python -m fire matmul/cuda_fp8.py measure` for Llama 70B training shapes.
`python -m fire matmul/cuda_fp8.py prof_matmul $m $k $n --repeats $r`
for user-provided shapes.
"""
from statistics import mean, median, stdev

import torch
from torch import nn


class SMM(nn.Module):
    """
    This class profiles multiple GEMMs because for small GEMMs,
    the host overhead is greater than the computation overhead,
    resulting in inaccurate measurements that appear too slow.
    """
    def __init__(
            self,
            m: int,
            k: int,
            n: int,
            device,
            rowwise: bool,
            use_fast_accum: bool,
            repeats: int = 1,
            fp8_config: str = "E4M3",
    ):
        super().__init__()
        dtype = dict(E4M3=torch.float8_e4m3fn, E5M2=torch.float8_e5m2).get(fp8_config)
        self.m1s = [torch.randn(size=(m, k), device=device).to(dtype=dtype) for _ in range(repeats)]
        self.m2s = [torch.randn(size=(n, k), device=device).to(dtype=dtype) for _ in range(repeats)]
        self.use_fast_accum = use_fast_accum
        if rowwise:
            s1 = torch.ones(m, 1, device=device)
            s2 = torch.ones(1, n, device=device)
        else:
            s1 = s2 = torch.ones(1, device=device)
        self.s1 = s1
        self.s2 = s2

    @torch.compile(fullgraph=True, dynamic=False)
    def forward(self):  # Equivalent to einsum("bmk,bnk->mn")
        outs = list()
        kwargs = dict(out_dtype=torch.bfloat16, use_fast_accum=self.use_fast_accum)
        for m1, m2 in zip(self.m1s, self.m2s, strict=True):
            outs.append(torch._scaled_mm(m1, m2.T, self.s1, self.s2, **kwargs))
        return outs


@torch.inference_mode()
def prof_matmul(
        m: int,
        k: int,
        n: int,
        rowwise: bool,
        use_fast_accum: bool,
        warmup_steps: int = 32,
        num_steps: int = 256,
        repeats: int = 1,  # Increase this value for small matrices below 4Kx4K.
        fp8_config: str = "E4M3",
) -> None:
    smm = SMM(
        m=m,
        k=k,
        n=n,
        device="cuda",
        rowwise=rowwise,
        use_fast_accum=use_fast_accum,
        repeats=repeats,
        fp8_config=fp8_config,
    )
    for _ in range(warmup_steps):  # Warmup
        smm()

    tics = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    tocs = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    for i in range(num_steps):
        tics[i].wait()  # Prevent asynchronous launches.
        tics[i].record()
        smm()
        tocs[i].record()
    torch.cuda.synchronize()

    mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]

    # Using the exact matmul flops equation.
    tfps = [1e-9 * repeats * m * n * (2 * k - 1) / ms for ms in mss]
    print(
        f"({m:5}x{k:5})x({k:5}x{n:5}) E4M3 {use_fast_accum=}: "
        f"Mean {mean(tfps):6.1f} TFLOPS, "
        f"Median {median(tfps):6.1f} TFLOPS, "
        f"Min {min(tfps):6.1f} TFLOPS, "
        f"Max {max(tfps):6.1f} TFLOPS, "
        f"STDEV {stdev(tfps):4.1f} TFLOPS, "
        f"H100 MFU: {mean(tfps) / 1978.9 * 100:4.1f}%"
    )


def measure(
        warmup_steps: int = 32,
        num_steps: int = 256,
        fp8_config: str = "E4M3",  # Only E4M3 works for now.
        rowwise: bool = False,
        use_fast_accum: bool = True,

) -> None:
    torch._dynamo.reset()  # Clear compilation cache.
    torch._dynamo.config.cache_size_limit = 64
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
            rowwise=rowwise,
            use_fast_accum=use_fast_accum,
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            repeats=r,
            fp8_config=fp8_config,
        )
