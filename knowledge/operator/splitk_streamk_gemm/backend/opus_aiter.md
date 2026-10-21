---
title: "OPUS Split-K: workspace, cluster launch and fused reduction"
kind: implementation_knowledge
operator: splitk_streamk_gemm
backend: opus
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# OPUS Split-K: workspace, cluster launch and fused reduction

The [OPUS exact-id API](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/README.md) distinguishes the selected GEMM instance from its workspace and reduction plan. The native `.cuh` bodies expose several schedules that were absent from the previous source scan.

| Native variant | What changes |
| --- | --- |
| [gfx1250 plain-grid TDM workspace](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh) | Uses a plain grid despite the filename; publishes Split-K partials to workspace and runs a separate reduction. |
| [gfx1250 cluster-launch workspace](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh) | A distinct cluster-launch pipeline and workspace schedule. |
| [gfx1250 cluster-launch fused reduction](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh) | Reduces partials within one kernel using a cluster barrier; still uses partial workspace. |
| [gfx950 flat-M Split-K](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh) | A different architecture-specific A16W16 decomposition. |

The fused cluster implementation gives each Split-K workgroup part of a tile and lets the final workgroup consume published partials. Adjacent M tiles can share a TDM-multicast B tile. Its reduction order, workspace dtype and bias handling are part of numerical correctness. One device launch does not mean there is no temporary storage.

Use [the launch plan](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/launch_plan.py) and [reduction launcher](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/opus_gemm/include/gfx1250/splitk_reduce_launch_gfx1250.cuh) to determine complete execution cost. Compare identical M/N/K, exact kernel id, split count and workspace ownership; validate partial rounding and bias as well as the final matrix.

## Evidence and validation

- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_opus_a16w16_gemm.py)
- [op_tests/test_opus_a16w16_registry.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_opus_a16w16_registry.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
