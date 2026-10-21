# act_and_mul_silu_gelu

5 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Activation and gated elementwise products — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Activation and gated elementwise products — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Activation and gated elementwise products — AITER gluon](backend/gluon_aiter.md) |
| hip | [Activation and gated elementwise products — AITER hip](backend/hip_aiter.md) |
| triton | [Activation and gated elementwise products — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: act_and_mul_silu_gelu — overview
kind: operator_overview
operator: act_and_mul_silu_gelu
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/activation.py
  - /sgl-workspace/aiter/aiter/ops/triton/activation.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu
```

</details>

# act_and_mul_silu_gelu  (`y = act(x[:d]) · x[d:]`, the gated MLP/MoE activation)

## TL;DR
This is the **gated activation** at the heart of every SwiGLU/GeGLU MLP and MoE expert: split the
up/gate projection `[M, 2d]` into two halves, apply SiLU/GeLU to the gate half, and **elementwise-multiply**
by the up half → `[M, d]`. It is **memory-bound** (reads 2d, writes d) and runs **once per layer** (×n_experts
in MoE). The dominant optimization is **fusion**: fold it into the up/gate GEMM epilogue, and/or fuse the
**fp8/fp4 quant** of its output into the same kernel so the down-proj GEMM reads quantized input.

## Math contract
Input `x[M, 2d]` (concatenated gate||up from the fused up/gate Linear). `d = x.shape[-1]//2`.
- **SiLU/SwiGLU**: `y = silu(x[:, :d]) · x[:, d:]`, `silu(z)=z·sigmoid(z)`.
- **GeLU/GeGLU**: `y = gelu(x[:, :d]) · x[:, d:]` (exact erf or tanh approx `gelu_tanh`).
- `mul_and_silu` variant swaps which half is gated. dtype: bf16/fp16 in, **fp32 act compute**, bf16/fp16
  out (or fp8/fp4 + scale in the quant-fused variant). Layout: contiguous last dim → split is a stride.

## Shape regimes (hidden 4096/5120/8192, inter 14336/17408/...)
- **prefill**: `M = tokens` (1k–64k), `2d = intermediate·2` (28k–35k). Many rows → fills the chip.
- **decode**: `M = batch` (1..256) → few rows, latency-bound; persistent grid.
- **MoE**: per-expert `M = tokens routed to expert`; the activation is inside the fused-MoE stage-1
  epilogue (gate/up GEMM → act_and_mul → stage-2 down GEMM). See [fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md).

## Where it matters (Amdahl)
Standalone act_and_mul is **1–3%** GPU time, but as a separate pass it forces a `[M, 2d]` read + `[M, d]`
write between two GEMMs. Fusing it into the GEMM epilogue removes that whole HBM round-trip; fusing the
output quant halves the down-proj input traffic. In MoE it's part of the kernel that AMD's FlyDSL rewrite
drove to **+162% throughput** on Kimi-K2.5 (the `silu_and_mul_fq` fused activation+quant).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (CK/asm `silu_and_mul` + flydsl fused quant) | [backends/aiter.md](backend/aiter.md) |
| triton | 🟢 sota (aiter act+quant Triton kernels) | [backends/triton.md](backend/triton.md) |
| hip | 🟢 sota (vLLM `silu_and_mul` HIP) | [backends/hip.md](backend/hip.md) |
| vllm_kernels | 🟢 sota (HIP act.cu + AITER wiring) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |
| flydsl | 🟢 sota (SiLU·mul + fused MXFP4/MXFP8 quant `silu_and_mul_fq`) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
`+up/gate GEMM` epilogue ([gemm_epilogue_fused](../gemm_epilogue_fused/README.md), [dense_gemm](../dense_gemm/README.md)); `+fp8/fp4 dynamic quant` of the output
→ [fused_norm_quant](../fused_norm_quant/README.md) / [quant_dequant_fp8](../quant_dequant_fp8/README.md) / [quant_fp4_mxfp](../quant_fp4_mxfp/README.md) (aiter `scaled_silu_and_mul`,
`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`; flydsl `silu_and_mul_fq`); inside fused-MoE
stage-1 ([fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md)). See [fusion.md](fusion.md).

## Numerics
fp32 act compute; exact-erf vs tanh GeLU; fp8/fp4 quant accuracy gate. See [numerics.md](numerics.md).

## How to bench
Isolated `silu_and_mul(out, x)` at `(M, 2d)`; fp64 oracle; median ≥3 reps. e2e: toggle the GEMM-epilogue
fusion / output-quant fusion. See [tuning.md](tuning.md).

## Sources
- aiter activation ops (`silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `scaled_silu_and_mul`): `/sgl-workspace/aiter/aiter/ops/activation.py`.
- aiter Triton act+quant (`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`): `/sgl-workspace/aiter/aiter/ops/triton/activation.py`.
- flydsl fused activation+quant `silu_and_mul_fq`: `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py`.
- vLLM HIP `silu_and_mul`/`gelu_and_mul`: https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/act_and_mul_silu_gelu/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
