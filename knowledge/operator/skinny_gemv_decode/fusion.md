---
title: skinny_gemv_decode — fusion
kind: technique
operator: skinny_gemv_decode
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
---

# skinny_gemv_decode — fusion

## TL;DR
> Because the kernel is bandwidth bound, **any fusion that avoids re-reading activations/weights is pure
> win**: fold bias/activation and fp8 dequant into the epilogue, and fuse upstream norm+quant so the input
> never round-trips. (Weights still dominate bandwidth, but eliminating activation passes still helps.)

## Fusion neighbors
- **Bias + activation epilogue** after the split-K reduction → [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).
- **Upstream norm+quant**: rmsnorm→fp8 quant feeding the decode GEMM input →
  [../fused_norm_quant/overview.md](../fused_norm_quant/README.md).
- **fp8 dequant scale** in epilogue → [../scaled_quant_gemm/fusion.md](../scaled_quant_gemm/fusion.md).
- **Decode MoE**: tiny per-expert M makes the grouped GEMM behave like many skinny GEMMs →
  [../grouped_gemm_moe/overview.md](../grouped_gemm_moe/README.md).
- Core decomposition technique: [../splitk_streamk_gemm/fusion.md](../splitk_streamk_gemm/fusion.md)
  (apply epilogue once after reduction, never per K-partition).

## Caveat
- With split-K, bias/activation must be applied only after the cross-partition reduction (not per
  partition) — same rule as splitk_streamk_gemm.

## Sources
- AITER decode/skinny fused paths: https://github.com/ROCm/aiter

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/skinny_gemv_decode/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
