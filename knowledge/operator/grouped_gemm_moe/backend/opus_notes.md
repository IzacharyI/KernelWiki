---
title: "OPUS MoE: stage-1 schedules and stage-2 output strategies"
kind: implementation_knowledge
operator: grouped_gemm_moe
backend: opus
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# OPUS MoE: stage-1 schedules and stage-2 output strategies

[The native MoE source guide](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/README.md) and [stage-1 binding](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/moe_stage1_a8w4.py) describe gfx950 A8W4 expert computation. The device pipeline headers distinguish more than one grouped-GEMM schedule.

| Stage / variant | Source and meaningful difference |
| --- | --- |
| Stage 1 group-split | [Group-split pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/include/gfx950/a8w4/stage1/opus_moe_stage1_a8w4_pipeline_group_split_gfx950.cuh); route-tile remapping and group decomposition. |
| Stage 1 paired K-waves | [Paired K-wave pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/include/gfx950/a8w4/stage1/opus_moe_stage1_a8w4_pipeline_pair_kwave_gfx950.cuh); explicitly requires cross-K-wave reduction in its traits. |
| Stage 2 direct output | [Decode pipeline](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh); selected instances atomically accumulate into token output. |
| Stage 2 route output | The same family can emit BF16 or MXFP8 per-route workspace, followed by [route reduction](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/include/gfx950/opus_moe_stage2_route_output_reduce_gfx950.cuh). |

Follow [stage-2 dispatch](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_moe/include/gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh) for the mapping from public algorithm id and effective intermediate dimension to the compiled specialization. Padding is not an independent public kernel id. The source guide also retains a private BF16 stage-2 body; it is not exposed as a general Python/fused-MoE path and should not be presented as a deployable candidate.

## Why the output strategy matters

Direct accumulation avoids a route-output reduction but changes write contention and accumulation order. Route output adds workspace traffic and a reduction; MXFP8 route output additionally changes intermediate precision and scale storage. `AITER_OPUS_STAGE2_FP8=0` provides a BF16 route-workspace comparison for relevant tuned entries while preserving their selected GEMM geometry.

Evaluate both expert stages, sorting metadata, activation semantics and final combine. Include skewed token distributions and empty experts; a uniform grouped-GEMM microbenchmark can conceal the serving workload's critical path.

## Evidence and validation

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
