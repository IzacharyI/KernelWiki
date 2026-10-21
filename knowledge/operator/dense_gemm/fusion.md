---
title: dense_gemm — fusion
kind: technique
operator: dense_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/optimizing-with-composable-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
---

# dense_gemm — fusion

## TL;DR
The biggest fusion wins are **GEMM epilogue fusions** (bias/act/residual/quant) — folding the elementwise
pass into the GEMM C write-out removes a full [M,N] memory round-trip. The dedicated card for this is
[operators/gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md); this note covers what dense_gemm fuses with and why.

## Fusion neighbors
- **+bias**: free in the epilogue (CShuffle `D = A·Bᵀ + bias`). Note `bias` is part of aiter's lookup key.
- **+act (silu/gelu) `act_and_mul`**: the up/gate MLP GEMM fuses the gated activation, removing a
  separate [M, inter*2] elementwise pass → see [operators/act_and_mul_silu_gelu/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/act_and_mul_silu_gelu/README.md) and
  [operators/gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).
- **+residual add**: fold the skip connection into the epilogue, killing a [M,N] read+write.
- **+fp8/fp4 quant**: quantize C in the epilogue (scaled write-out) so the next GEMM consumes fp8
  directly → see [operators/scaled_quant_gemm/overview.md](../scaled_quant_gemm/README.md) and [operators/quant_dequant_fp8/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_dequant_fp8/README.md).
- **norm→GEMM**: RMSNorm+quant before the GEMM is fused on the *producer* side →
  see [operators/fused_norm_quant/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md).

## Why it pays (Amdahl)
GEMM is compute-bound; the appended epilogue is essentially free vs the matmul FLOPs, but the *separate*
elementwise kernel it replaces is a full memory-bound pass over [M,N]. On MLP up/gate (N up to 34816)
that pass is large, so fusing `act_and_mul` is one of the better cheap wins.

## How it's expressed per backend
- **CK**: `DeviceGemmMultipleD_Xdl_CShuffle` with a `CDEElementwise` op (bias/act/residual via the
  CShuffle stage) → `languages/composable_kernel/...`.
- **triton/tilelang/flydsl**: append the epilogue inside the kernel after the MFMA accumulation, before
  the store; use `OPTIMIZE_EPILOGUE=1` to avoid the 512B Tagram hotspot on the C write.
- **aiter/hipBLASLt**: bias + scale are supported epilogue args on the tuned path.

## Pitfalls
- Fusing changes the lookup key (bias/act/scale flags) → must capture/tune the *fused* shape, or the
  tuned DB misses (same failure mode as bias mismatch in [backends/aiter.md](backend/aiter.md)).
- Over-fusing rare paths increases kernel-variant count and tuning time for little Amdahl gain.

## Sources
- CShuffle epilogue / MultipleD: ROCm "Optimizing with Composable Kernel".
- Tagram/epilogue hotspot, OPTIMIZE_EPILOGUE: ROCm workload guide.
- bias/scale as epilogue args on live path: `ROCm/aiter@HEAD:aiter/tuned_gemm.py`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/dense_gemm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
