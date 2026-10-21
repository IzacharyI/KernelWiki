---
title: rope — fusion neighbors
kind: technique
operator: rope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py
  - /sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# rope — fusion

RoPE sits at the **attention entry** and fuses heavily with everything around it. Standalone it's an extra
Q/K round-trip; fused it's near-free.

## 1. QK-norm + RoPE + KV-cache write + quant  (the Qwen3 mega-fusion)
The dominant production fusion: apply Q/K RMSNorm, then RoPE, then write K/V to the paged KV cache, with
optional fp8 quant — all in one kernel.
- `fused_qk_norm_rope_cache_quant` (aiter, standard RoPE).
- `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` (mrope/3D variant → [mrope](../mrope/README.md)).
SGLang cites this fused QKNorm+RoPE+KV-set kernel as a significant Qwen3/Qwen3-VL per-layer win (#18466).
Cross-link [rmsnorm](../rmsnorm/README.md), [fused_norm_quant](../fused_norm_quant/README.md), [kv_cache_quant](../kv_cache_quant/README.md), [paged_kv_copy](../paged_kv_copy/README.md).

## 2. RoPE + KV-cache write
`rope_cached_*` and `fused_kv_cache` (Triton `fusions/fused_kv_cache.py`, `fused_bmm_rope_kv_cache.py`)
apply RoPE to K and write it to the cache in one pass. `rope_cached_positions_*` take positions inline.
Cross-link [paged_kv_copy](../paged_kv_copy/README.md).

## 3. QKV-split + RoPE
`fused_qkv_split_qk_rope` (Triton): split the fused QKV projection output and apply RoPE to Q,K in one
kernel. Cross-link [dense_gemm](../dense_gemm/README.md) (QKV proj).

## 4. inside attention prefill
Some FMHA kernels apply RoPE on-the-fly at the QKᵀ load. Cross-link [attention_prefill_fmha](../attention_prefill_fmha/README.md).

## Fusion table
| form | impl | folds in |
|---|---|---|
| QK-norm+RoPE+KV+quant | `fused_qk_norm_rope_cache_quant` | norm, rope, kv-write, fp8 |
| mrope variant | `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | + 3D rope → [mrope](../mrope/README.md) |
| RoPE + KV write | `rope_cached_*`, `fused_kv_cache` | kv-write |
| QKV-split + RoPE | `fused_qkv_split_qk_rope` | qkv split |

## torch.compile
vLLM `VLLM_ROCM_USE_AITER_TRITON_ROPE` and the RoPE custom-op registration keep the fused kernel through
Inductor. See [backends/vllm_kernels/aiter_integration](../../general/libraries/vllm_kernels/aiter_integration.md).

## Sources
- QK-norm+RoPE+KV+quant fusion: `/sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- RoPE+KV / QKV-split+RoPE (Triton): `/sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py`, `rope/fused_qkv_split_qk_rope.py`.
- Qwen3 fused QKNorm+RoPE+KV win: https://github.com/sgl-project/sglang/issues/18466.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/rope/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
