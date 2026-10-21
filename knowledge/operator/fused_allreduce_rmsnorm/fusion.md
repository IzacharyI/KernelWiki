---
title: fused_allreduce_rmsnorm — fusion
kind: technique
operator: fused_allreduce_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/vllm-project/vllm/issues/26768
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# fused_allreduce_rmsnorm — fusion

This operator **is** the fusion. The dimensions are: how much of the chain you fold, and AR-epilogue vs
SP-rewrite.

## The fusion patterns (vLLM, `enable_fi_allreduce_fusion`)
| pattern | chain folded |
|---|---|
| `AllReduceRMSNormPattern` | AR → RMSNorm |
| `AllReduceFusedAddRMSNormPattern` | AR → (+residual) → RMSNorm |
| `AllReduceFusedRMSNormStaticQuantFP8Pattern` | AR → RMSNorm → fp8 static quant |
| `AllReduceFusedAddRMSNormStaticQuantFP8Pattern` | AR → (+residual) → RMSNorm → fp8 static quant |

These are torch.compile passes (`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`) that match the
chain and emit one kernel — folding 2–4 ops (collective + add + norm + quant) into the AR epilogue.

## The two realizations
1. **AR-epilogue fusion**: keep the all-reduce, fuse add/norm/quant into its epilogue (the vLLM patterns).
2. **SP rewrite, one kernel**: aiter `reduce_scatter_rmsnorm_quant_all_gather` (`fused_pipeline_kernel`)
   does RS → RMSNorm (sharded) → fp8 quant → AG in a single Iris-based Triton kernel — the fully-fused SP
   collective. Pick this when norm-on-shard + RS/AG overlap wins.

## Sibling: qknorm + all-reduce
aiter's `qknorm_allreduce_fusion_kernel_2stage` (MiniMax-M2.x) fuses QK-norm with the AR (grid-strided, no
80-token cap) — **~10–15% prefill TPS at TP=2/4** (AMD-reported). Same family: fold a norm onto the
collective.

## Fusion neighbors
- collective half: [allreduce](../allreduce/README.md) / [reduce_scatter](../reduce_scatter/README.md) + [allgather](../allgather/README.md).
- norm half: [rmsnorm](../rmsnorm/README.md) / [fused_add_rmsnorm](../fused_add_rmsnorm/README.md).
- quant epilogue: [fused_norm_quant](../fused_norm_quant/README.md) → feeds an fp8 [scaled_quant_gemm](../scaled_quant_gemm/README.md) / [fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md).

## What is NOT yet fused
- A single kernel that fuses the AR+norm with the **next GEMM** (so the normed+quanted activation never hits
  global memory before the GEMM) — AsyncTP overlaps but doesn't fuse into the GEMM kernel.

## Cross-links
[allreduce](../allreduce/README.md) · [reduce_scatter](../reduce_scatter/README.md) · [allgather](../allgather/README.md) · [rmsnorm](../rmsnorm/README.md) · [fused_norm_quant](../fused_norm_quant/README.md) ·
[`backends/aiter/integration.md`](../../general/libraries/aiter/integration.md) ·
[`backends/mori_rccl/rccl_tuning.md`](../../general/libraries/mori_rccl/rccl_tuning.md).

## Sources
- vLLM fusion patterns / passes: https://docs.vllm.ai/en/latest/design/fusions/ ; https://github.com/vllm-project/vllm/issues/26768
- aiter fused SP kernel + qknorm+AR: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`; https://github.com/ROCm/aiter/releases

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/fused_allreduce_rmsnorm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
