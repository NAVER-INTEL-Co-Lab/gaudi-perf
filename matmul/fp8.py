"""
Profile FP8 scaled GEMM throughput at different configurations.

Example command for (4096 x 16384) x (16384 x 8192) matmul:
PT_HPU_WEIGHT_SHARING=0 python -m fire matmul/fp8.py prof_matmul 4096 16384 8192
"""
import warnings
from statistics import mean, stdev, median

import torch
from torch import nn, Tensor
import habana_frameworks.torch as ht


class FP8GEMM(nn.Module):
    def __init__(self, s1: Tensor, s2: Tensor, si1: Tensor, si2: Tensor):
        super().__init__()
        self.scale1 = nn.Parameter(s1)
        self.scale2 = nn.Parameter(s2)
        self.scale_inv1 = nn.Parameter(si1)
        self.scale_inv2 = nn.Parameter(si2)

    def forward(
            self,
            x1,
            x1_fp8,
            rowwise1,
            do_cast1,
            use_sr1,
            x2,
            x2_fp8,
            rowwise2,
            do_cast2,
            use_sr2,
            fp8_dtype,
    ):
        if do_cast1:
            self.scale1 = self.scale1[:, None] if rowwise1 else self.scale1
            x1_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
                input=x1,
                scale=self.scale1,
                stochastic_rounding=use_sr1,
                is_amax=False,
                dtype=fp8_dtype,
                scale_shape=None,
            )
        if do_cast2:
            scale2 = self.scale2[:, None] if rowwise2 else self.scale2
            x2_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
                input=x2,
                scale=scale2,
                stochastic_rounding=use_sr2,
                is_amax=False,
                dtype=fp8_dtype,
                scale_shape=None,
            )

        return torch.ops.hpu.fp8_gemm_v2(
            A=x1_fp8,
            trans_A=False,
            B=x2_fp8,
            trans_B=True,
            D=None,  # Not sure what this is.
            out_dtype=torch.bfloat16,
            A_scale_inv=self.scale_inv1,
            B_scale_inv=self.scale_inv2,
            bias=None,
            accumulate=False,  # What does this do?
            B_scale_shape=None,
        )


class FP8GEMMS(nn.Module):
    def __init__(
            self,
            s1: Tensor,
            s2: Tensor,
            si1: Tensor,
            si2: Tensor,
            repeats: int,
            x1,
            x1_fp8,
            x2,
            x2_fp8,
    ):
        super().__init__()
        self.fp8_gemm = FP8GEMM(s1=s1, s2=s2, si1=si1, si2=si2)
        self.repeats = repeats
        self.x1s = x1 if x1 is None else [x1.clone() for _ in range(repeats)]
        self.x2s = x2 if x2 is None else [x2.clone() for _ in range(repeats)]
        self.x1_fp8s = None if x1_fp8 is None else [x1_fp8.clone() for _ in range(repeats)]
        self.x2_fp8s = None if x2_fp8 is None else [x2_fp8.clone() for _ in range(repeats)]

    def forward(
            self,
            rowwise1,
            do_cast1,
            use_sr1,
            rowwise2,
            do_cast2,
            use_sr2,
            fp8_dtype,
    ):
        out = 0
        for i in range(self.repeats):
            out += self.fp8_gemm(
                x1=self.x1s[i] if do_cast1 else None,
                x1_fp8=self.x1s_fp8[i] if not do_cast1 else None,
                rowwise1=rowwise1,
                do_cast1=do_cast1,
                use_sr1=use_sr1,
                x2=self.x2s[i] if do_cast2 else None,
                x2_fp8=self.x2s_fp8[i] if not do_cast2 else None,
                rowwise2=rowwise2,
                do_cast2=do_cast2,
                use_sr2=use_sr2,
                fp8_dtype=fp8_dtype,
            )

@torch.inference_mode()
def prof_matmul(
        m: int,
        k: int,
        n: int,
        fp8_type: str = "E4M3",
        hw_scale1: bool = True,
        hw_scale2: bool = True,
        rowwise1: bool = False,
        rowwise2: bool = False,
        do_cast1: bool = False,
        do_cast2: bool = False,
        use_sr1: bool = False,
        use_sr2: bool = False,
        hpu_graph: bool = False,
        num_steps: int = 256,
        warmup_steps: int = 32,
        repeats: int = 1,  # Increase this value for small matrices below 4Kx4K.
):
    ht.core.hpu_set_inference_env()
    configs = {k: v for k, v in locals().items() if not k.startswith("_")}
    if (use_sr1 and not do_cast1) or (use_sr2 and not do_cast2):
        warnings.warn("Invalid configuration.")
        return
    device = torch.device("hpu")
    fp8_dtype = dict(E4M3=torch.float8_e4m3fn, E5M2=torch.float8_e5m2).get(fp8_type)
    x1 = torch.rand(size=(m, k), device=device, dtype=torch.bfloat16)
    x2 = torch.rand(size=(n, k), device=device, dtype=torch.bfloat16)

    s1 = 1.0 if hw_scale1 else 1.5  # 1.0 is hardware accelerated, 1.5 is not.
    s1 = [s1 for _ in range(m)] if rowwise1 else s1
    s1 = torch.tensor(s1).to(device)
    if not do_cast1:
        x1_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
            input=x1,
            scale=s1[:, None] if rowwise1 else s1,
            stochastic_rounding=False,
            is_amax=False,
            dtype=fp8_dtype,
            scale_shape=None,
        )
    else:
        x1_fp8 = None

    s2 = 1.0 if hw_scale2 else 1.5  # 1.0 is hardware accelerated, 1.5 is not.
    s2 = [s2 for _ in range(n)] if rowwise2 else s2
    s2 = torch.tensor(s2).to(device)
    if not do_cast2:
        x2_fp8, _ = torch.ops.hpu.cast_to_fp8_v2(
            input=x2,
            scale=s2[:, None] if rowwise2 else s2,
            stochastic_rounding=False,
            is_amax=False,
            dtype=fp8_dtype,
            scale_shape=None,
        )
    else:
        x2_fp8 = None

    si1 = 1 / s1[:, None] if rowwise1 else 1 / s1
    si2 = 1 / s2[None, :] if rowwise2 else 1 / s2

    steps = num_steps + warmup_steps
    tics = [ht.hpu.Event(enable_timing=True) for _ in range(steps)]
    tocs = [ht.hpu.Event(enable_timing=True) for _ in range(steps)]
    fp8_gemm = FP8GEMMS(s1=s1, s2=s2, si1=si1, si2=si2, repeats=repeats,
                        x1=x1, x1_fp8=x1_fp8, x2=x2, x2_fp8=x2_fp8).to(device)
    ht.core.hpu_inference_initialize(fp8_gemm, mark_only_scales_as_const=True)
    if hpu_graph:  # HPU graphs make the run slower. I do not know why.
        fp8_gemm = ht.hpu.wrap_in_hpu_graph(fp8_gemm)

    for i in range(num_steps + warmup_steps):
        tics[i].wait()  # Prevent asynchronous launches.
        tics[i].record()
        out = fp8_gemm(
            # x1=x1,
            # x1_fp8=x1_fp8,
            rowwise1=rowwise1,
            do_cast1=do_cast1,
            use_sr1=use_sr1,
            # x2=x2,
            # x2_fp8=x2_fp8,
            rowwise2=rowwise2,
            do_cast2=do_cast2,
            use_sr2=use_sr2,
            fp8_dtype=fp8_dtype,
        )
        tocs[i].record()
        ht.core.mark_step()
    ht.hpu.synchronize()

    tics = tics[warmup_steps:]
    tocs = tocs[warmup_steps:]
    mss = [tic.elapsed_time(toc) for tic, toc in zip(tics, tocs, strict=True)]
    # Using the exact equation for matrix multiplication FLOP counts.
    tfps = [1e-9 * repeats * m * n * (2 * k - 1) / ms for ms in mss]
    print(
        *[f"{k}: {v}," for k, v in configs.items()],
        f"Mean: {mean(tfps):.1f} TFLOPS,",
        f"Median: {median(tfps):.1f} TFLOPS,",
        f"Min: {min(tfps):.1f} TFLOPS,",
        f"Max: {max(tfps):.1f} TFLOPS,",
        f"STDEV: {stdev(tfps):.2f} TFLOPS",
        f"Gaudi v2 MFU: {mean(tfps) / 865 * 100:4.1f}%, "
        f"Gaudi v3 MFU: {mean(tfps) / 1835 * 100:4.1f}%"
    )
