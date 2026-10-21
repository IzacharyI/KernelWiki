---
title: "OPUS BMM: exact-id BF16 and MXFP8 variants"
kind: implementation_knowledge
operator: batched_gemm
backend: opus
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# OPUS BMM: exact-id BF16 and MXFP8 variants

[The OPUS registry](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/README.md) exposes A16W16 BMM on gfx942/gfx950/gfx1250 and a distinct gfx950 MXFP8 BMM family. These must not be inferred from a non-batched GEMM's support table.

## BF16 exact-id BMM

`opus_bmm` accepts batch-first matrices and caller-owned output. The caller resolves an exact registered `kid` and checks its BMM and workspace contract. The pinned source explicitly has no high-level `batched_gemm_bf16_OPUS` wrapper. Start at [the native BMM API](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/opus_bmm.h) and [implementation dispatch](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/opus_bmm.cu).

## MXFP8 BMM variants

The `a8w8_mxscale_bmm` family has 45 registered ids within the sparse global range 8000--8653 at this snapshot. It consumes FP8 payloads plus E8M0 scales, and produces BF16 or FP32 output. The public layout name is `mxscale_bmm`.

| Source | Implementation role |
| --- | --- |
| [MX-scale pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh) | Matrix/scaling dataflow for the main family. |
| [Compact runtime pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_bmm_pipeline_mxscale_compact_gfx950.cuh) | Tile-dependent segment scheduling and runtime M/N/K/batch. |
| [Launchers](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_bmm_launchers_a8w8_mxscale_gfx950.cuh) | Instance launch and shape contracts. |
| [Python launch plan](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/launch_plan.py) | Buffer sizes, layouts and optional Split-K workspace. |

The compact body's assumptions include positive M below 2048 and batch up to 16, plus instance-specific alignment and divisibility requirements. Those are constraints of that body, not the whole OPUS BMM family. Compare batch scheduling, scale movement and reduction cost with the selected CK/FlyDSL/Gluon BMM path.

## Evidence and validation

- [op_tests/test_opus_a8w8_bmm.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_opus_a8w8_bmm.py)
- [op_tests/test_opus_a8w8_interface.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_opus_a8w8_interface.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
