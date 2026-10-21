---
title: reduce_scatter on HIP / Triton-Iris — SOTA card
kind: sota_card
operator: reduce_scatter
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/reduce_scatter.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  - https://github.com/ROCm/iris
---

# reduce_scatter × HIP / Triton-Iris (GPU-initiated)

## TL;DR
> The custom reduce-scatter path is **GPU-initiated comm**: aiter's Triton `reduce_scatter`
> (`_reduce_scatter_kernel`) and the **fused RS+RMSNorm+quant+AG** SP kernel. This is the seam to fuse RS
> with the norm/all-gather (the SP collective in one kernel) or to author an xGMI P2P RS with custom
> overlap. RCCL is the default standalone RS; this is the fusion/authoring path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| fused RS+RMSNorm+quant+AG | `.../comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | gfx942/950; bf16/fp16/fp8 | — (the SP collective, one kernel) | SP rewrite of all-reduce |
| aiter Triton `reduce_scatter` (Iris) | `aiter/ops/triton/comms/reduce_scatter.py` | gfx942/950 | — | fusing RS into a Triton kernel |
| HIP xGMI P2P RS | `languages/hip_cpp/` + IPC | gfx942/950 | — | bespoke RS/overlap |

## Config space / knobs
- Iris: symmetric heap, push/atomic over xGMI, persistent-grid, grid = CU count. Needs `iris` + symmetric
  init.
- HIP P2P: IPC handles, atomic-accumulate (`-munsafe-fp-atomics`) for the reduction, SDMA for the scatter.
- The fused kernel: norm epsilon/axis, optional fp8 quant stage (`_quantize_fp8_stage`).

## Numerics / parity
fp32 accumulate; reduction-order deltas benign; the fused kernel's fp8 stage is a gate (fnuz on gfx942). See
[numerics.md](../numerics.md).

## Integration (rebind seam)
aiter Triton comms under `aiter/ops/triton/comms/`; the fused SP kernel is the drop-in for AR+norm in an SP
layout. Edit the Triton/HIP source for a custom RS.

## Pitfalls & anti-patterns
- Iris experimental — benchmark vs RCCL.
- Atomic-accumulate contention on the reduction — partition by shard.
- Wrong shard boundaries → SP non-equivalence (greedy parity catches it).

## How to verify
rocprof confirms the Triton/Iris kernel ran; numeric vs fp32 reference; SP equivalence (greedy parity);
bandwidth vs RCCL (value is fusion, not standalone speed).

## Alternatives / cross-links
[mori_rccl.md](rccl.md) (default) · [allgather](../../allgather/README.md) · [fused_allreduce_rmsnorm](../../fused_allreduce_rmsnorm/README.md) ·
[`languages/triton_amd/`](../../../general/backend/triton_amd/overview.md) · [overview.md](../README.md).

## Sources
- aiter Triton reduce_scatter + fused SP kernel: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/reduce_scatter.py`, `.../fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- Iris: https://github.com/ROCm/iris

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/reduce_scatter/backends/hip.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
