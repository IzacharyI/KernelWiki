# quant_int8

0 current AITER implementation references; 0 historical experience cards.

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: quant_int8 — overview
kind: operator_overview
operator: quant_int8
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8, bf16, fp16]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/smoothquant.h
  - https://arxiv.org/abs/2211.10438
```

</details>

# quant_int8  (`x_i8 = round(x / s) [+ zp]`, SmoothQuant W8A8)

## TL;DR
INT8 W8A8 (8-bit weights *and* activations) with **SmoothQuant** is the classic pre-FP8 quant recipe and
still the right choice on **CDNA1/2 (no FP8 HW)** and for INT8-MFMA paths (INT32 accumulate). The trick
that makes INT8 activations work is **SmoothQuant**: migrate the activation outliers into the weights via
a per-channel smoothing factor `s_smooth`, so activations become quantization-friendly. Granularity is
**per-token** (activations) × **per-channel** (weights). On CDNA3/4 FP8 usually wins on accuracy at the
same throughput, so INT8 is mostly for older HW or INT8-native deployments.

## Math contract
- **Quant (symmetric)**: `s = amax / 127`; `x_i8 = round_to_nearest(x / s)` clamped to [-128, 127],
  INT32 accumulate in the GEMM, dequant `* s_a * s_w`.
- **Quant (asymmetric, azp)**: `x_i8 = round(x / s) + zp` with zero-point `zp = round(-128 - min/s)`;
  needs an azp correction term in the GEMM epilogue (vLLM `azp` path).
- **SmoothQuant**: choose per-channel `s_smooth = max(|X|)^α / max(|W|)^(1-α)`; apply
  `X̂ = X / s_smooth`, `Ŵ = W · s_smooth` offline so both quantize cleanly (arXiv 2211.10438).
- aiter `smoothquant_fwd(out, input, x_scale[1,n], y_scale[m,1])`: applies the per-channel smooth scale
  `x_scale` and emits per-token `y_scale` in one kernel.

## Scale granularity
| | activations | weights |
|---|---|---|
| **SmoothQuant default** | **per-token** dynamic (`[M,1]`) | **per-channel** static (`[N]`) |
| coarser | per-tensor | per-tensor |
| symmetric vs asymmetric | symmetric usual; azp for skewed | symmetric |

## Shape regimes
- prefill: per-token amax over `hidden` for `[M=tokens, hidden]`, M 1k–16k.
- decode: skinny M; launch-bound → fuse into norm.
- weights: quantized once offline (per-channel, static).

## Where it matters (Amdahl)
Same as FP8: the standalone quant is small, but it gates the INT8 GEMM (the Amdahl head). On CDNA1/2 where
FP8 MFMA does not exist, INT8 is the *only* sub-16-bit matrix-core path.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| vllm_kernels | 🟢 sota (own int8 HIP, symmetric + azp) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |
| aiter | 🟢 sota (smoothquant + pertoken, MoE int8) | [backends/aiter.md](backend/aiter.md) |
| hip | 🟢 sota (editable int8 HIP source) | [backends/hip.md](backend/hip.md) |
| triton | 🟡 competitive (fused smoothquant MoE int8) | [backends/triton.md](backend/triton.md) |

## Fusion neighbors
RMSNorm + int8 quant, SmoothQuant per-channel scale folded into the cast, GEMM dequant epilogue
(`* s_a * s_w` + azp correction), MoE int8 grouped GEMM. → [fusion.md](fusion.md), [operators/fused_norm_quant](../fused_norm_quant/README.md),
[operators/scaled_quant_gemm](../scaled_quant_gemm/README.md), [operators/fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md).

## Numerics
INT32 accumulate, round-to-nearest (`nearbyint`, FE_TONEAREST), symmetric vs asymmetric, SmoothQuant α →
[numerics.md](numerics.md).

## How to bench
Isolated: `dynamic_scaled_int8_quant(out, x, scale[, azp])` over `[M, hidden]`; oracle = round-trip
error + INT8 GEMM result vs bf16. e2e: INT8 linear vs bf16/FP8, gate on tok/s AND task accuracy.

## Sources
- vLLM int8 quant (symmetric + azp, nearbyint): `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- aiter pertoken/smoothquant: `ROCm/aiter@a6bb49937:aiter/ops/quant.py`, `csrc/include/smoothquant.h`.
- SmoothQuant: https://arxiv.org/abs/2211.10438

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/quant_int8/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
