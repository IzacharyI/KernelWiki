---
title: batched_gemm — fusion
kind: technique
operator: batched_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
---

# batched_gemm — fusion

## TL;DR
The single most important "fusion" for batched GEMM is **into attention**: QKᵀ and PV batched matmuls
get fused with softmax into a fused-MHA kernel, so the [B,H,S,S] scores never hit HBM. After that, the
same epilogue fusions as dense (bias/act/scale) apply per batch.

## Fusion neighbors
- **→ attention (FMHA)**: QKᵀ·softmax·PV fused — the dominant case; standalone batched GEMM should only
  survive where attention isn't fused. See [operators/attention_prefill_fmha/overview.md](../attention_prefill_fmha/README.md),
  [operators/gqa_mqa_attention/overview.md](../gqa_mqa_attention/README.md), [operators/mla_attention/overview.md](../mla_attention/README.md).
- **+bias / +act epilogue (per batch)**: same CShuffle-style epilogue as dense →
  [operators/gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).
- **+fp8 quant epilogue (per batch)**: scaled write-out so the next batched matmul consumes fp8 →
  [operators/scaled_quant_gemm/overview.md](../scaled_quant_gemm/README.md).
- **→ grouped GEMM**: when per-batch shapes differ (MoE experts), fuse into a single grouped launch
  instead of padded batched → [operators/grouped_gemm_moe/overview.md](../grouped_gemm_moe/README.md),
  [operators/fused_moe_grouped_gemm/overview.md](../fused_moe_grouped_gemm/README.md).

## Why it pays
The attention fusion removes the giant [B,H,S,S] scores tensor (memory-bound) — far larger than any
epilogue win. Per-batch epilogue fusions remove a [B,M,N] elementwise pass, same Amdahl logic as dense.

## How it's expressed per backend
- **CK**: `DeviceBatchedGemmMultipleD_Xdl_CShuffle` with a `CDEElementwise` epilogue per batch.
- **triton**: append epilogue after MFMA before the per-batch store; `OPTIMIZE_EPILOGUE=1`.
- For attention fusion, use the dedicated FMHA kernels, not a fused batched-GEMM.

## Pitfalls
- Fused shape/flags change the tuning key → tune the fused variant.
- Padding variable per-batch shapes into one batched kernel wastes FLOPs — prefer grouped GEMM.

## Sources
- CK batched MultipleD epilogue: ROCm "Optimizing with Composable Kernel".
- Epilogue args on live path: `ROCm/aiter@HEAD:aiter/tuned_gemm.py`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/batched_gemm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
