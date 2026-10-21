---
title: mrope — fusion neighbors
kind: technique
operator: mrope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# mrope — fusion

Same fusion family as [rope](../rope/README.md) — the attention-entry mega-fusion, specialized for 3D positions.

## 1. QK-norm + mRoPE + KV-cache write + quant (+ shuffle)
aiter `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`: Q/K RMSNorm → 3D mRoPE (per `mrope_section`) →
KV-cache write (paged, with positions) → optional fp8 quant → KV-layout shuffle, all in one kernel. The
VLM analog of the Qwen3 `fused_qk_norm_rope_cache_quant` win (#18466). Cross-link [rmsnorm](../rmsnorm/README.md),
[fused_norm_quant](../fused_norm_quant/README.md), [kv_cache_quant](../kv_cache_quant/README.md), [paged_kv_copy](../paged_kv_copy/README.md), [layout_shuffle](../layout_shuffle/README.md).

## 2. mRoPE + KV write
The cached mRoPE variants apply mRoPE to K and write to the cache in one pass (the 3D analog of
`rope_cached_*`). Cross-link [paged_kv_copy](../paged_kv_copy/README.md).

## 3. inside VLM attention prefill
mRoPE applied at the QKᵀ load for image/video tokens. Cross-link [attention_prefill_fmha](../attention_prefill_fmha/README.md).

## Fusion table
| form | impl | folds in |
|---|---|---|
| QK-norm+mRoPE+KV+quant+shuffle | `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | norm, 3D rope, kv-write, fp8, shuffle |
| mRoPE + KV write | cached mrope variant | kv-write |

## Sources
- aiter fused QK-norm+mrope+KV+quant+shuffle: `/sgl-workspace/aiter/aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- analogous Qwen3 RoPE fusion win: https://github.com/sgl-project/sglang/issues/18466.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/mrope/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
