# quant_fp4_mxfp

5 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Packed low-bit and microscaling formats — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Packed low-bit and microscaling formats — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Packed low-bit and microscaling formats — AITER gluon](backend/gluon_aiter.md) |
| hip | [Packed low-bit and microscaling formats — AITER hip](backend/hip_aiter.md) |
| triton | [Packed low-bit and microscaling formats — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: quant_fp4_mxfp — overview
kind: operator_overview
operator: quant_fp4_mxfp
gens: [gfx950]
dtypes: [fp4_e2m1, fp6_e2m3, fp6_e3m2, mxfp4, mxfp6, mxfp8]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
```

</details>

# quant_fp4_mxfp  (MXFP4 / MXFP6 — 32-element block + E8M0 scale)

## TL;DR
Microscaling (MX): cast bf16/fp16 to **FP4 (E2M1)** or **FP6 (E2M3/E3M2)** where every **32 consecutive
elements share one E8M0 (8-bit, exponent-only) scale**. This is the **CDNA4-only** (gfx950) low-bit path
— the block-scaled MFMA `v_mfma_scale_f32_*_f8f6f4` applies the E8M0 scale after the dot product, running
FP4 *and* FP6 at **10 PF** (the FP4 rate). The whole point of block scaling: a single per-tensor FP4 scale
collapses (FP4 has ~3 bits of resolution), but a per-32-element scale lets each block self-normalize, so
MXFP4 **weight-only** quant is viable. **On CDNA3 (MI300X) there is NO FP4/FP6 HW** — MXFP4 there is
software-simulated (dequant-on-the-fly), and vLLM `FP4BMM` *crashes* gfx942. → [hardware/cdna4_mi350](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna4_mi350).

## Math contract
For `x[M,N]`, group size **32** along the quantized dimension:
- per 32-element block: `block_amax = max(|x_block|)`; **E8M0 scale** `s = f32_to_e8m0(block_amax /
  FP4_MAX)` where `FP4_MAX = 2^floor(log2(6)) = 4` (E2M1 max is 6; the kernel uses the power-of-2 max).
- `x_fp4 = f32_to_mxfp4(x_block / e8m0_to_f32(s))`; two FP4 values pack into one `uint8`
  (`dtypes.fp4x2`).
- **Layout**: `A(M,K) → fp4 (M, K//2)`, `scale (M, K//32)`. For the `tl.dot_scaled` RHS, pack along K
  (`pack_dim=0`): `B(K,N) → fp4 (K//2, N)`, `scale (K//32, N)`.
- E8M0: value `2^(scale-127)`; `scale=127` ⇒ ×1; `E=255` reserved NaN; range `2^-127 … 2^127`.

## Scale shuffle (hardware layout)
The MFMA wants the E8M0 scales in a **shuffled** layout. aiter's `per_1x32_f4_quant_hip(..., shuffle=True)`
allocates the scale padded to `((m+255)//256*256, ((n+31)//32+7)//8*8)` and runs `e8m0_shuffle` so Ax/Bx
match `v_mfma_scale_*`'s operand map. Wrong shuffle → silent corruption ([numerics.md](numerics.md)).

## FP4 vs FP6 (same speed, different accuracy)
FP6 (E2M3 more mantissa / E3M2 more range) runs at the **same 10 PF** as FP4 — choosing FP6 costs
*accuracy headroom, not throughput*. Use MXFP4 for the largest weight tensors; **MXFP6** (or mixed
MXFP4/MXFP6) when FP4 is too lossy. AMD reports MXFP4 near-lossless on very large models (DeepSeek-R1) but
noticeable degradation on small/mid models, where MXFP6/mixed wins at the same FLOPs.

## Shape regimes
- **weight-only (most common)**: quantize weights once offline (Quark `w_mxfp4_a_mxfp4`, group 32);
  activations bf16 or MXFP — the GEMM is the consumer.
- **activation MXFP4 (dynamic)**: `dynamic_mxfp4_quant` casts activations at runtime (group 32, fixed by
  spec); vLLM supports only dynamic activation MXFP4.

## Where it matters (Amdahl)
Halves weight memory vs FP8 (4 bits + 0.25 scale ≈ 4.25 b/elt) → bigger models / more KV headroom, and on
CDNA4 doubles GEMM throughput (10 PF vs 5 PF FP8). The quant itself is bandwidth-bound; the win is the
GEMM ([operators/scaled_quant_gemm](../scaled_quant_gemm/README.md)) and the memory footprint.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (HIP + triton MX quant, shuffle) | [backends/aiter.md](backend/aiter.md) |
| triton | 🟢 sota (`dynamic_mxfp4_quant`, dot_scaled prep) | [backends/triton.md](backend/triton.md) |
| hip | 🟢 sota (`dynamic_per_group_scaled_quant_fp4`) | [backends/hip.md](backend/hip.md) |
| ck | 🟢 sota (mxfp4/mxfp6 GEMM + cast) | [backends/ck.md](backend/ck.md) |

## Fusion neighbors
RMSNorm + MXFP4 (`fused_rms_mxfp4_quant`), act_and_mul + MXFP4 (`fused_reduce_act_mul_and_mxfp4_quant`),
MoE sort + MXFP4 (`fused_dynamic_mxfp4_quant_moe_sort`), GEMM block-scaled epilogue. → [fusion.md](fusion.md),
[operators/fused_norm_quant](../fused_norm_quant/README.md), [operators/scaled_quant_gemm](../scaled_quant_gemm/README.md), [operators/fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md).

## Numerics
E8M0 block scale, group=32, scale rounding to power-of-2, FP4 vs FP6, accuracy gates, CDNA3 simulation →
[numerics.md](numerics.md), [hardware/cdna4_mi350](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna4_mi350), [hardware/shared/dtype_numerics](../../general/hardware/shared/dtype_numerics.md).

## How to bench
Isolated: `per_1x32_f4_quant(x, shuffle=True)` over `[M, hidden]`; oracle = round-trip per-block error +
MXFP4 GEMM vs bf16. e2e (gfx950 only): MXFP4 linear vs FP8/bf16, gate on tok/s AND task accuracy. On
gfx942: simulation only — no HW speedup.

## Sources
- OCP MX spec (group 32, E8M0, MXFP4/6/8): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- aiter MX quant (`per_1x32_f4_quant`, shuffle, e8m0): `ROCm/aiter@a6bb49937:aiter/ops/quant.py`.
- Triton `dynamic_mxfp4_quant` (group 32 fixed): `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/quant.py`.
- CDNA4 block-scaled MFMA, FP6@FP4 rate: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/quant_fp4_mxfp/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
