---
title: skinny_gemv_decode — numerics
kind: technique
operator: skinny_gemv_decode
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
---

# skinny_gemv_decode — numerics

## TL;DR
> Same fp32-accumulate rules as dense GEMM; the only added seam is the **split-K reduction** (the skinny
> path almost always splits K), which makes atomic-reduce results non-deterministic — use workspace
> reduction when you need reproducibility.

## Considerations
- **Accumulate fp32**; cast to bf16/fp16 at the epilogue.
- **Split-K reduction**: atomic-add (fast, non-deterministic order) vs workspace (deterministic). Avoid
  bf16/fp16 atomics → accumulate/atomic in fp32. See [../splitk_streamk_gemm/numerics.md](../splitk_streamk_gemm/numerics.md).
- **fp8 weights**: apply per-tensor/per-block scales after the dot →
  [../scaled_quant_gemm/numerics.md](../scaled_quant_gemm/numerics.md). fp8 weight quant in decode is a
  common quality risk because errors accumulate token-over-token — gate with a generation-quality metric.
- **GEMV vs MFMA path parity**: a VALU GEMV and an MFMA skinny kernel must agree with the dense reference
  within fp tolerance; verify both against the same oracle.

## Parity gate
- Oracle = dense fp32 reference for the same (X,W). bf16: tight relative error. fp8: relative-error band +
  downstream task gate (PPL/MMLU delta). For deterministic CI, use workspace reduction.

## Sources
- AITER skinny/scaled paths: https://github.com/ROCm/aiter

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/skinny_gemv_decode/numerics.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
