---
title: context_parallel_attention — fusion
kind: technique
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# context_parallel_attention — fusion

## The fusion that defines CP: comm ⊗ compute
CP's performance is about **overlapping** the collective with the local attention, and **fusing** comm
into the kernel where possible:

| fused/overlapped piece | where | benefit |
|---|---|---|
| **KV-block P2P send/recv ⊗ local FA tile** | ring loop | hide XGMI transfer under compute (the whole point) |
| **GPU-initiated comm in-kernel** | aiter Iris (`comms/iris.py`) | Triton kernel issues all-gather/reduce-scatter inline → tighter overlap than a separate RCCL call |
| **fused reduce-scatter + RMSNorm + quant + all-gather** | aiter `comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | collapses a comm+norm+quant chain (TP/SP epilogue) into one pass |
| **incremental LSE-merge** | per received KV block | merge as blocks arrive; don't buffer all then merge |
| **split-KV transfer for PD-disagg** | prefill→decode KV handoff | overlap KV transfer with the next ring step (SGLang proposal) |

## What does NOT fuse
- The local FA tile and the LSE-merge stay as the standard FA kernels ([attention_prefill_fmha](../attention_prefill_fmha/README.md)); CP
  wraps them, it doesn't replace the matmul.
- All-to-all (Ulysses) does a single local full-seq FA with no cross-rank merge — fewer fusion seams but
  bigger messages; pick ring vs all-to-all by message-size vs overlap tradeoff.

## Cross-links
- Collectives: [all_to_all_dispatch_combine](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/all_to_all_dispatch_combine/README.md) · [allgather](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/allgather/README.md) · [reduce_scatter](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/reduce_scatter/README.md) ·
  [fused_allreduce_rmsnorm](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_allreduce_rmsnorm/README.md) (related fused comm).
- Local attention: [attention_prefill_fmha](../attention_prefill_fmha/README.md) · [chunked_prefill](../chunked_prefill/README.md) (extend/prefix path).
- KV: [paged_kv_copy](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/paged_kv_copy/README.md) · [kv_cache_quant](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/kv_cache_quant/README.md).
- Languages/backends: `triton_amd` · `aiter` · `sglang_kernels` · `mori_rccl` (EP all-to-all).

## Sources
- aiter fused comm + Iris GPU-initiated primitives: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/` (on-box) ; https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- SGLang split-KV transfer + zigzag ring: https://github.com/sgl-project/sglang/issues/22223

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/context_parallel_attention/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
