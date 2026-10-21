---
title: scaled_quant_gemm — fusion
kind: technique
operator: scaled_quant_gemm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/aiter
---

# scaled_quant_gemm — fusion

## TL;DR
> The high-value fusions are at the **quant boundaries**: fuse input quantization (norm→quant) before the
> GEMM and output quantization after it, so activations never materialize in high precision between ops.
> The scale pipeline itself is "fused" by staging scales through LDS alongside the tile loads.

## Fusion neighbors
- **Upstream norm+quant**: rmsnorm/layernorm producing fp8/fp4 + per-block scales directly into the GEMM
  input → [../fused_norm_quant/overview.md](../fused_norm_quant/README.md),
  [../quant_fp4_mxfp/overview.md](../quant_fp4_mxfp/README.md),
  [../quant_dequant_fp8/overview.md](../quant_dequant_fp8/README.md).
- **Epilogue bias/activation + output quant**: dequant accumulator, add bias, activation, re-quantize
  output to fp8/fp4 with new block scales → [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).
- **MoE**: scaled grouped GEMM with per-expert/per-block scales →
  [../grouped_gemm_moe/fusion.md](../grouped_gemm_moe/fusion.md).
- **KV-cache quant** shares the fp8 quant machinery → [../kv_cache_quant/overview.md](../kv_cache_quant/README.md).

## Scale-pipeline note
- Scales must be staged Global→LDS(re-layout)→LDS-read to feed scaled MFMA; there is no register-direct
  scale feed (Gluon tutorial). Treat this staging as part of the kernel, pipelined with A/B loads.

## Sources
- Gluon GEMM tutorial (scale pipeline): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- AITER fused quant GEMM paths: https://github.com/ROCm/aiter

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/scaled_quant_gemm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
