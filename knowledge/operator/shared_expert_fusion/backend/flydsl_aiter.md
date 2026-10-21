---
title: "FHMoE: routed and shared experts in one execution path"
kind: implementation_knowledge
operator: shared_expert_fusion
backend: flydsl
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# FHMoE: routed and shared experts in one execution path

[The public fused_moe entry](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/fused_moe.py) dispatches to [FHMoE](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/fhmoe.py) when shared-expert weights/scales or a shared-expert id are supplied. This is a concrete AITER implementation for the shared-expert family that previously had only historical GEAK notes.

| Variant | Routed weights | Shared weights | Main distinction |
| --- | --- | --- | --- |
| MXFP4/FP8 FHMoE | MXFP4 | FP8 payload with per-32 E8M0 scales | Heterogeneous routed/shared storage. |
| MXFP8/FP8 FHMoE | MXFP8 | FP8 payload with per-32 E8M0 scales | Separate tuned rows and explicit preshuffle validation. |

The [stage wrappers](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/flydsl/fhmoe.py) call [shared-expert stage builders](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/flydsl/kernels/fhmoe.py). The shared tensors stay separate from routed weights; the contract reserves a dummy final routed row with `shared_expert_id == E - 1`.

## Constraints that change the comparison

The pinned contract requires gfx950, SiLU, per-1x32 quantization, contiguous preshuffled weights/scales and all four shared weight/scale tensors. Expert-mask and expert-bias dispatch are unsupported. Validate the complete wrapper contract before treating FHMoE as a replacement for a generic routed-plus-shared MoE.

`clamp_shared` changes model semantics: true clamps both routed and shared SiLU, while false leaves the shared branch unclamped. The stage-1 compilation identities differ. Compare matching activation semantics and include routing, both expert stages, reduction and shared contribution in the workload timing.

Related communication fusion: [the direct BF16 reduce-scatter/shared-add body](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py) is a separate compute/communication boundary; its presence does not imply FHMoE selects it automatically.

## Evidence and validation

- [op_tests/test_fhmoe.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_fhmoe.py)
- [op_tests/tuners/tune_fhmoe.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/tuners/tune_fhmoe.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
