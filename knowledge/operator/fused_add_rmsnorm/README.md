# fused_add_rmsnorm

2 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| gluon | [Residual addition and RMS normalization — AITER gluon](backend/gluon_aiter.md) |
| triton | [Residual addition and RMS normalization — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: fused_add_rmsnorm — overview
kind: operator_overview
operator: fused_add_rmsnorm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/14959
```

</details>

# fused_add_rmsnorm  (`r' = x + r; y = rmsnorm(r')` in one kernel)

## TL;DR
This is the **dominant on-serving-path form of RMSNorm**: a transformer block does `h = h + sublayer(...)`
and the next norm reads `h`, so the residual-add is fused into the norm. One read of `x`+`residual`, one
write of `residual_out`+`y` — instead of add(write h)→norm(read h). It is **memory-bound** and runs **2×
per layer**. This op IS the [rmsnorm](../rmsnorm/README.md) serving path; read [rmsnorm](../rmsnorm/README.md) first, this is the fused variant.

## Math contract
Inputs `x[M,N]` (sublayer output), `residual_in[M,N]`. Outputs: `residual_out = x + residual_in` (the new
residual stream, written for the next block), `y = rmsnorm(residual_out)·γ`. dtype: bf16/fp16 in, **fp32**
sum-of-squares accumulate, bf16/fp16 out; `residual_out` kept in the IO dtype (it's the running residual).
**Both outputs are written** — that's the key: the residual must persist for the skip connection.

## Shape regimes
Identical to [rmsnorm](../rmsnorm/README.md): prefill `M=tokens` (row-per-program), decode `M=batch` (persistent
`min(M,num_sms)`), `N ∈ {4096,5120,8192}`. The extra `residual_in` read + `residual_out` write doubles the
traffic vs plain rmsnorm — making the fusion (vs two kernels) even more worthwhile.

## Where it matters (Amdahl)
It's the **2-per-layer** norm, so ~2–4% GPU time on a dense LLM, and the fusion saves the `h` round-trip on
every one. Combined further with fp8 quant ([fused_norm_quant](../fused_norm_quant/README.md)) it's part of SGLang's **1–6% e2e** Qwen3
gain. Engaged by `VLLM_ROCM_USE_AITER_RMSNORM=1` (vLLM PR #14959 wired `rmsnorm2d_fwd_with_add`).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (CK/asm `rmsnorm2d_fwd_with_add` / `add_rmsnorm`) | [backends/aiter.md](backend/aiter.md) |
| triton | 🟢 sota (`_fused_add_rmsnorm_kernel`) | [backends/triton.md](backend/triton.md) |
| hip | 🟢 sota (vLLM `fused_add_rms_norm_kernel`) | [backends/hip.md](backend/hip.md) |
| vllm_kernels | 🟢 sota (HIP + AITER wiring) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |

## Fusion neighbors
This is itself a fusion. Stack further: `+fp8/int8 dynamic quant` → `rmsnorm2d_fwd_with_add_dynamicquant`
/ `add_rmsnorm_quant` (residual+norm+quant triple) → [fused_norm_quant](../fused_norm_quant/README.md); `+all-reduce` (TP) →
[fused_allreduce_rmsnorm](../fused_allreduce_rmsnorm/README.md). See [fusion.md](fusion.md).

## Numerics
fp32 accumulate over `residual_out`; γ fp32-promote; the add happens in IO dtype then the Σx² in fp32.
See [numerics.md](numerics.md).

## How to bench
`python3 op_tests/test_rmsnorm2d.py` with the `_with_add` path; oracle = `rmsnorm(x+r)` fp64; e2e A/B
toggling `VLLM_ROCM_USE_AITER_RMSNORM`.

## Sources
- aiter `rmsnorm2d_fwd_with_add` / `add_rmsnorm` / `_ck`: `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`.
- Triton `_fused_add_rmsnorm_kernel`: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- vLLM HIP `fused_add_rms_norm_kernel`: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu.
- vLLM AITER with_add integration: https://github.com/vllm-project/vllm/pull/14959.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/fused_add_rmsnorm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
