---
title: splitk_streamk_gemm — fusion
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://arxiv.org/abs/2301.03598
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
---

# splitk_streamk_gemm — fusion

## TL;DR
> Fusion happens in the **reduction/fix-up epilogue**: once partial sums are combined, fold bias /
> activation / dequant-scale there so the result needs no extra pass — but only after the cross-partition
> reduction completes (epilogue must run once per output element, not once per partition).

## Fusion neighbors
- **Bias + activation epilogue**: applied after the final reduction → [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).
- **Dequant / output quant scale**: for fp8/mxfp GEMMs, apply scales in the epilogue after combining
  partials → [../scaled_quant_gemm/fusion.md](../scaled_quant_gemm/fusion.md).
- **Decode small-M**: the decode GEMV path often uses split-K internally (e.g. wvSplitK-style) →
  [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/README.md).

## Caveat
- Do **not** apply bias/activation per K-partition (split-K) — it would be added `SPLIT_K` times. Apply
  only in the final reduction step. With atomic-add reduction, bias must be initialized into the output
  buffer once (or added in a separate finalize), never inside the per-partition atomic.

## Sources
- Stream-K fix-up/epilogue: https://arxiv.org/abs/2301.03598
- Triton split-K epilogue placement: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/splitk_streamk_gemm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
