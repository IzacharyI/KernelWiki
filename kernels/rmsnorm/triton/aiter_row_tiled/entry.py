# SPDX-License-Identifier: MIT
"""Forward-only, contiguous RMSNorm for AITER's large-M/small-N regime."""

from functools import lru_cache
import json
import math
from pathlib import Path

import torch
import triton

if __package__:
    from .kernel import _rmsnorm_kernel_large_m_small_n
else:
    from kernel import _rmsnorm_kernel_large_m_small_n


@lru_cache(maxsize=1)
def _configs():
    return json.loads(Path(__file__).with_name("launch.json").read_text())


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Return x * rsqrt(mean(x**2) + eps) * weight in x.dtype.

    Inputs are contiguous, on the same ROCm GPU, with matching FP16/BF16 dtype.
    x is [M,N], weight is [N], M > 8192 and 1 <= N <= 2048. No backward is provided.
    """
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("expected x[M,N] and weight[N]")
    if not x.is_cuda or not torch.version.hip or x.device != weight.device:
        raise ValueError("inputs must be on the same ROCm GPU")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError("x and weight must have matching FP16 or BF16 dtype")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("this source bundle implements forward inference only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    m, n = x.shape
    if m <= 8192 or not 1 <= n <= 2048:
        raise ValueError("this implementation requires M > 8192 and 1 <= N <= 2048")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
    if arch not in _configs():
        raise ValueError(f"no launch configuration for {arch}")
    config = _configs()[arch]
    block_n = triton.next_power_of_2(n)
    block_m = max(8, min(16384 // block_n, 32))
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_kernel_large_m_small_n[(triton.cdiv(m, block_m),)](
            x, out, weight, None, m, n, eps,
            x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n,
            NUM_WARPS=config["num_warps"], NUM_STAGES=config["num_stages"],
            num_warps=config["num_warps"], num_stages=config["num_stages"],
        )
    return out
