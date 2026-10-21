---
title: "Native HK MLA: v3.2 and v4.0 decode pipelines"
kind: implementation_knowledge
operator: mla_attention
backend: hip
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Native HK MLA: v3.2 and v4.0 decode pipelines

[The HK MLA bindings](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/attention.py) expose separate `hk_mla_v32_decode_fwd` and `hk_mla_v40_decode_fwd` entries with architecture dispatch. The device bodies live in `.cuh` files, which the earlier inventory did not cover.

| Body | Distinction |
| --- | --- |
| [gfx950 v3.2 m16x4](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh) | Smaller warp-group schedule for the v3.2 FP8 path. |
| [gfx950 v3.2 m16x8](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh) | Alternate warp-group decomposition. |
| [MI3xx v3.2 body](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_m16x8_fp8_fp8.cuh) | Separate architecture route; inspect the binding's arch guard. |
| [gfx950 v4.0 m16x4](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x4_fp8bf16_fp8bf16_gen1.cuh) | Mixed FP8 NoPE and BF16 RoPE input planes. |
| [gfx950 v4.0 m16x8](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh) | Separate warp assignment and KV-tile orchestration for the mixed-format path. |

The [v4.0 shared stage implementation](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/hk_mla_v40_fwd_decode_gen1_common.cuh) makes register ownership explicit across QK, softmax and PV. Its [buffer managers](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/mla/hk/hk_mla_v40_buffer_managers_gen1.cuh) are part of the pipeline, not interchangeable cache-copy helpers.

## Validation boundary

V3.2 and V4.0 are different data/attention contracts. Do not compare them by replacing only a kernel name. Validate NoPE/RoPE planes, scale decoding, QK/V dimensions, head counts, partial outputs and LSE. Use the wrapper's architecture and shape checks before selecting m16x4 or m16x8.

The repository's [v4.0 reference setup](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_mla_v40_persistent.py) explicitly documents reference/metadata coverage with limited kernel comparison; it is not evidence that every HK variant was tested. Kernel-level validation must invoke the intended HK entry and compare its complete result to the matching reference.

## Evidence and validation

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
