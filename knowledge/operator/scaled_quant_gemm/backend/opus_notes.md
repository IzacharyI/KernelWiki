---
title: "OPUS FP8 GEMM: scale and preshuffle families"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: opus
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# OPUS FP8 GEMM: scale and preshuffle families

The [public registry](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/README.md) separates plain FP8, FP8 with block scales, preshuffled block-scale GEMM and MXFP8 BMM. Their output types and scale contracts differ. A native header with a compatible-looking dtype is not sufficient evidence of a public dispatch route.

| Public family at this snapshot | Contract |
| --- | --- |
| gfx950 `a8w8` | Plain FP8 inputs, no scale tensors, FP32 output. |
| gfx950 `a8w8_blockscale` | Plain FP8 inputs, two FP32 scale tensors, FP32 output. |
| gfx942 `a8w8_blockscale_bpreshuffle` | Preshuffled weights, FP32 scale tensors, BF16 output. |
| gfx950 `a8w8_mxscale_bmm` | FP8 payload/E8M0 scales and BF16 or FP32 output; see [BMM](../../batched_gemm/backend/opus_aiter.md). |

The [gfx942 block-scale pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx942/a8w8/opus_gemm_pipeline_a8w8_blockscale_bpreshuffle.cuh) exposes the weight-layout and scale dataflow beneath the public adapter.

## Additional native schedule bodies

The source also contains gfx950 MX-scale/preshuffle [one-wave](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_wave1_gfx950.cuh), [eight-wave](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_wave8_gfx950.cuh), [persistent eight-wave](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_wave8_persist_gfx950.cuh) and [flat-M Split-K](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh) bodies. Treat these as implementation source until an exact registered id and reachable adapter are established for the requested case; this page does not advertise every header as an additional supported public kernel.

Record scale granularity, packing, output precision and exact id with measurements. Include live activation conversion, weight-layout preparation when repeated, and Split-K reduction in end-to-end projection cost.

## Evidence and validation

- [op_tests/test_opus_a8w8_interface.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_opus_a8w8_interface.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
