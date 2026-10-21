# SPDX-License-Identifier: MIT
"""FP32-accumulation mathematical reference, including epsilon."""

import torch


def rmsnorm_reference(x, weight, eps=1e-6):
    x32 = x.float()
    inverse_rms = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return (x32 * inverse_rms * weight.float()).to(x.dtype)
