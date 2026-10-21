# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Extracted from ROCm/aiter e9995854abbd935f0190b19abebe9938b6b90cca.
# Kernel body and naming helper are unchanged; imports are local to this bundle.

import triton
import triton.language as tl

if __package__:
    from .kernel_repr import make_kernel_repr
else:
    from kernel_repr import make_kernel_repr


_rmsnorm_kernel_large_m_small_n_repr = make_kernel_repr(
    "_rmsnorm_kernel_large_m_small_n",
    [
        "BLOCK_M",
        "BLOCK_N",
        "NUM_WARPS",
        "NUM_STAGES",
    ],
)


@triton.jit(repr=_rmsnorm_kernel_large_m_small_n_repr)
def _rmsnorm_kernel_large_m_small_n(
    X,
    Y,
    W,
    RSIGMA,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_off = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int64)
    n_off = tl.cast(tl.arange(0, BLOCK_N), tl.int64)

    mask_m = m_off < M
    mask_n = n_off < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        X + m_off[:, None] * stride_xm + n_off[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(W + n_off, mask=mask_n, other=0.0).to(tl.float32)

    x = tl.where(mask, x, 0.0)
    sum_sq = tl.sum(x * x, axis=1)
    var = sum_sq / N
    rsigma = tl.math.rsqrt(var + eps)

    y = x * rsigma[:, None] * w[None, :]
    tl.store(
        Y + m_off[:, None] * stride_ym + n_off[None, :] * stride_yn,
        y.to(Y.dtype.element_ty),
        mask=mask,
    )

    if RSIGMA is not None:
        tl.store(RSIGMA + m_off, rsigma, mask=mask_m)
