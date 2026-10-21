# quant_dequant_fp8

6 current AITER implementation references; 8 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Quantization and dequantization — AITER aiter](backend/aiter_aiter.md) |
| asm | [Quantization and dequantization — AITER asm](backend/asm_aiter.md) |
| flydsl | [Quantization and dequantization — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Quantization and dequantization — AITER gluon](backend/gluon_aiter.md) |
| hip | [Quantization and dequantization — AITER hip](backend/hip_aiter.md) |
| triton | [Quantization and dequantization — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: quant_dequant_fp8 — overview
kind: operator_overview
operator: quant_dequant_fp8
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, bf16, fp16]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
```

</details>

# quant_dequant_fp8  (`x_fp8 = round(x / s)`, `x ≈ x_fp8 · s`)

## TL;DR
Cast bf16/fp16 activations (and weights) to **FP8** with a scale `s` so the matrix core can run the
GEMM at 2× (CDNA3) / FP8-rate (CDNA4) — and dequant on the way out. The single most important fact:
**FP8 is FNUZ on CDNA3 (gfx942) and OCP on CDNA4 (gfx950)** — different bias and saturation, so a
checkpoint and its quant kernel must agree on the dialect or you read garbage off by ~2×. The standalone
quant op is **memory-bound** (one pass over the tensor); the real lever is **fusing** it into the
producer (RMSNorm/act) or the consumer (GEMM epilogue) so the FP8 tensor never round-trips HBM →
[fusion.md](fusion.md), [operators/fused_norm_quant](../fused_norm_quant/README.md), [operators/scaled_quant_gemm](../scaled_quant_gemm/README.md).

## Math contract
For input `x[M,N]` (bf16/fp16) and target FP8 `e4m3`/`e5m2`:
- **Quant**: `s = amax(x_block) / FP8_MAX`; `x_fp8 = clamp(x / s, -FP8_MAX, FP8_MAX)` cast to FP8.
- **Dequant**: `x ≈ x_fp8.to(f32) * s`.
- Some kernels store the **reciprocal** scale (`is_scale_inverted` → multiply by `1/s`) to turn the hot
  per-element divide into a multiply (vLLM `scaled_fp8_conversion<true>`).
- `FP8_MAX` depends on dialect: OCP E4M3 **448**, FNUZ E4M3FNUZ **240** — but vLLM uses **224.0** on
  ROCm for *dynamic* quant (240 hurts accuracy; see [numerics.md](numerics.md)).

**Scale granularity** (the design axis that dominates accuracy/perf):
| granularity | scale shape for `x[M,N]` | when |
|---|---|---|
| **per-tensor static** | scalar (precomputed, calibration) | fastest; weights, well-behaved activations |
| **per-tensor dynamic** | scalar (amax this call) | activations without calibration |
| **per-token dynamic** | `[M,1]` (amax per row) | activations (per-token is the SmoothQuant-friendly default) |
| **per-block / per-group** | `[M, N/128]` or 1×128 / 1×32 | fine-grained (DeepSeek-style block fp8, group=128); 1×32 is the MX path → [operators/quant_fp4_mxfp](../quant_fp4_mxfp/README.md) |

## Shape regimes
- **prefill**: quantize the full activation `[M=tokens, hidden]` before each linear; `M` 1k–16k. Bandwidth
  ∝ tensor size; per-token amax is a row-reduction over `hidden`.
- **decode**: skinny `M=batch` (1..256); the quant is tiny but launch overhead matters → fuse.
- **weights**: quantized once offline (Quark) → per-tensor or per-channel/per-block, static.

## Where it matters (Amdahl)
A standalone quant pass is ~1–3% of GPU time but it **gates** the FP8 GEMM (the ~80% Amdahl head). The
win is not the quant kernel itself — it is enabling the FP8 matrix core and *removing* the pass by
fusion. Unfused, every linear pays an extra full-tensor read+write of the activation.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path: dispatches HIP/CK/triton quant) | [backends/aiter.md](backend/aiter.md) |
| vllm_kernels | 🟢 sota (own `scaled_fp8_quant` HIP, ROCm 224 cap) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |
| hip | 🟢 sota (the editable HIP source under aiter/vLLM) | [backends/hip.md](backend/hip.md) |
| triton | 🟡 competitive (portable, fusion-friendly) | [backends/triton.md](backend/triton.md) |
| asm | 🟡 (hand-tuned cast inside fused asm kernels) | [backends/asm.md](backend/asm.md) |

## Fusion neighbors
`rmsnorm + quant` (fused_rms_fp8_*_quant), `act_and_mul + quant`, GEMM **dequant epilogue**
(`gemm_a8w8` fuses the `*s` back), KV-store quant. → [fusion.md](fusion.md),
[operators/fused_norm_quant](../fused_norm_quant/README.md), [operators/scaled_quant_gemm](../scaled_quant_gemm/README.md), [operators/kv_cache_quant](../kv_cache_quant/README.md).

## Numerics
e4m3 vs e5m2 ranges, FNUZ↔OCP, amax/scale, 224-cap, stochastic rounding, accuracy gates →
[numerics.md](numerics.md), [hardware/shared/dtype_numerics](../../general/hardware/shared/dtype_numerics.md), [hardware/cdna4_mi350](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna4_mi350).

## How to bench
Isolated: time `dynamic_per_token_scaled_quant(out, x, scale)` over `[M, hidden]` at prefill/decode M;
oracle = round-trip max/rel error vs an fp32 reference cast (gate per [numerics.md](numerics.md)). e2e: measure the
FP8 *linear* (quant+GEMM+dequant) vs bf16, gate on tok/s delta AND task accuracy (gsm8k/mmlu), not byte
parity.

## Sources
- aiter quant entrypoints (`pertoken_quant`, `static/dynamic_per_tensor`, `dynamic_per_token_scaled_quant`):
  `ROCm/aiter@a6bb49937:aiter/ops/quant.py`.
- HIP quant kernels (`scaled_quant_impl`, `dynamic_per_token_scaled_quant`): `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu`.
- vLLM FP8 quant (224.0 ROCm cap, scale-inverted, per-token): `vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu`.
- FNUZ vs OCP, FP8 ranges: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/quant_dequant_fp8/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
