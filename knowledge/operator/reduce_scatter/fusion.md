---
title: reduce_scatter — fusion
kind: technique
operator: reduce_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# reduce_scatter — fusion

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **reduce-scatter + RMSNorm + fp8 quant + all-gather** | the entire SP collective in one kernel | aiter `reduce_scatter_rmsnorm_quant_all_gather.py` (`fused_pipeline_kernel`) | one fused kernel replaces AR+norm |
| **GEMM → reduce-scatter (AsyncTP)** | producer GEMM overlapped with the RS of its output | vLLM AsyncTP pass | hides RS under compute |
| **reduce-scatter → norm** (SP) | norm runs on the sharded (1/P) tensor | SP layout | less norm work + overlap |

## The SP collective as one kernel (the headline)
`reduce_scatter_rmsnorm_quant_all_gather` (`fused_pipeline_kernel` + `_quantize_fp8_stage`) is aiter's
single Triton kernel for the **sequence-parallel rewrite of all-reduce**: it reduce-scatters the partial,
RMSNorms the sharded result, optionally fp8-quantizes, then all-gathers — exactly the
`AR → RS + norm + AG` transform, fused. This is the most fused realization on AMD and the reason to prefer
the SP path on TP layers: the norm is cheaper (sharded) and the whole thing is one launch built on Iris
GPU-initiated comm. Ties [reduce_scatter](README.md) + [allgather](../allgather/README.md) + [fused_allreduce_rmsnorm](../fused_allreduce_rmsnorm/README.md) together.

## AsyncTP overlap
Without full fusion, the AsyncTP pass overlaps RS with the producing GEMM (the GEMM emits partials that RS
reduces while the next op starts). Pairs with `GPU_MAX_HW_QUEUES=2` / high-priority streams.

## Cross-links
[allgather](../allgather/README.md) · [fused_allreduce_rmsnorm](../fused_allreduce_rmsnorm/README.md) · [allreduce](../allreduce/README.md) (the op SP replaces) ·
[`backends/mori_rccl/rccl_tuning.md`](../../general/libraries/mori_rccl/rccl_tuning.md).

## Sources
- aiter fused SP kernel: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- SP / AsyncTP fusion passes: https://docs.vllm.ai/en/latest/design/fusions/

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/reduce_scatter/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
