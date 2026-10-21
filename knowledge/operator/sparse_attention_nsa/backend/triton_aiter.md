---
title: "Sparse attention: AITER triton"
kind: implementation_knowledge
operator: sparse_attention_nsa
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse attention — AITER triton

Sparse kernels consume selected pages, blocks or token indices. Selection granularity, duplicates, padding and causal masking affect the logical work independently of dense attention shapes.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/pa_decode_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py) · **Recent**

Source documentation: Sparse paged-decode attention over a unified or split KV pool with per-token paged indices. TODO: add details once API has settled

Symbols: [`_check_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L67), [`pa_decode_sparse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L78), [`_as_int32_contiguous_1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L459), [`_decode_num_splits_occ`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L465), [`_last_round_fills`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L484), [`_rounds_suit_32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L490), [`_dsv4_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L500), [`_staged_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L516), [`_launch_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L529), [`_pa_decode_sparse_gfx950_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py#L536).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `BLOCK_M`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pa_prefill_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py) · **Recent**

Source documentation: Sparse paged-prefill attention over two KV sources (prefix + extend) with per-head sink bias — gfx1250 (gluon) only. Exposes pa_prefill_sparse — grid (T, cdiv(H, BLOCK_H)), one token and BLOCK_H heads per CTA. Same grid as the decode kernel. No split-K: prefill fills the GPU via the token dimension. Gfx950(mla_gluon) and others(Triton _sparse_attn_prefill_kernel, they take a 1-D (kv_indices, kv_indptr) pair over one pool.

Symbols: [`_pinned_prefill_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py#L49), [`pa_prefill_sparse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py#L66), [`_prep_single_source`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py#L330), [`_as_index_contiguous_1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py#L366), [`_as_int32_contiguous_1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py#L374).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/qsa.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py) · **Recent**

Source documentation: QSA operators for index scoring, selection and sparse paged GQA.

Symbols: [`_prev_pow2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L18), [`_table_width_bucket`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L24), [`_kv_splits_heuristic`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L29), [`_kernel_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L48), [`_check_vector`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L74), [`qsa_paged_mqa_logits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L83), [`qsa_expand_block_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L215), [`qsa_sparse_paged_gqa`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py#L276).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`, `BLOCK_Q`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py) · **Recent**

Source documentation: DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4). Counterpart to the DSv4 sparse prefill forward. The op is the official V4 form: shared-KV GQA where K == V == kv is a single dense 512-wide tensor, RoPE already applied in place caller-side, scale 1/sqrt(512), attn_sink folded into the softmax denominator only, and topk_indices == -1 masked out.

Symbols: [`sparse_mla_bwd_dq`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L61), [`sparse_mla_bwd_dkv_interm_v4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L114), [`delta_v4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L166), [`build_inverted_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L190), [`dkv_gather_acc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L220), [`_BwdPlan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L244), [`_plan_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L292), [`_bwd_phases`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L420), [`sparse_mla_bwd_dsv4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_attention_dsv4_bwd.py#L482).

Parameters referenced in this file: `BLOCK_E`, `BLOCK_H`, `BLOCK_R`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py) · **Recent**

Source documentation: Sparse (top-k gathered) MLA attention. Everything after softmax_scale is keyword-only: kv_indptr and kv_indices are both flat int32, so a positional mix-up would type-check and return garbage. Supported KV cache formats. The format is inferred from kv_buffer's shape, dtype and kv_scale; each row gives what the caller has to pass. R is the QK width, kv_lora_rank + qk_rope_head_dim.

Symbols: [`_check_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L29), [`_cache_pointers`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L43), [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L88), [`_check_packed_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L99), [`_check_fp8_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L114), [`_check_lds_budget`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L147), [`_mla_num_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L173), [`_async_launch_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L189), [`_resolve_dot_precision`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L229), [`_dsv32_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L255), [`_classify_flat`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L259), [`_classify_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L286), [`_check_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L337), [`_check_index_stream`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L376), [`_forward_paged`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L396), [`sparse_mla_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py#L457).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/sparse_mla_dsv4_train.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py) · **Recent**

Symbols: [`_build_inverted_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py#L27), [`sparse_mla_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py#L65), [`sparse_mla_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py#L124), [`SparseMLADSV4Function`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py#L251), [`sparse_mla_dsv4_train`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py#L279).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_G`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py) · **Recent**

Source documentation: Sparse paged-decode attention kernels (split-K + per-token paged indices). Two-kernel decomposition of a flash-decode whose K range for each token is a gathered subset of a unified KV pool: _pa_decode_sparse : split-K main kernel. Grid (T, ceil(H/BLOCK_H), KV_SPLITS). Each program owns one token, one head-block, and one slice of the token's sparse K range; writes pre-sink (m, l, acc) partials.

Symbols: [`_pa_decode_sparse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py#L52), [`_pa_decode_sparse_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py#L283).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/qsa.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py) · **Recent**

Source documentation: QSA triton kernels for index scoring and sparse paged GQA.

Symbols: [`_qsa_paged_mqa_logits_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py#L42), [`_qsa_paged_mqa_logits_mfma_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py#L165), [`_expand_selected_groups`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py#L259), [`_qsa_sparse_paged_gqa_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py#L308).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py) · **Recent**

Symbols: [`_get_lds_limit`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L13), [`_pack_dense_prefix_to_ragged_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L36), [`_compute_topk_lens_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L68), [`_pack_global_topk_ragged_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L94), [`_compute_combined_lens_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L136), [`_combine_topk_swa_indices_ragged_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L164), [`_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L231), [`_prefill_prune_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L237), [`_get_prefill_autotune_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L251), [`_sparse_attn_prefill_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py#L294).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `BLOCK_SIZE`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4_bwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4_bwd.py) · **Recent**

Source documentation: Triton kernels for the DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4). The two memory-bound phases; the MFMA phases are Gluon and live in aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_attention_dsv4_bwd. _delta_v4_kernel delta = rowsum(O * dO) -- the standard flash-attention "o_dot_do" preamble. Streams the bf16 inputs and accumulates in fp32, so it moves exactly the working set.

Symbols: [`_delta_v4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4_bwd.py#L37), [`_bwd_dkv_gather_acc_v4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4_bwd.py#L66).

Parameters referenced in this file: `BLOCK_E`, `BLOCK_R`.

### `aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py) · **Recent**

Symbols: [`_sparse_mla_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py#L29), [`_bwd_dq_store_dp_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py#L149), [`_bwd_dkv_interm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py#L278), [`_bwd_dkv_gather_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py#L364).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_G`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py) · **Recent**

Symbols: [`cdiv_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py#L8), [`apply_softcap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py#L13), [`find_seq_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py#L21), [`_kernel_unified_attention_sparse_mla_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py#L60).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SIZE`.

### `aiter/ops/triton/attention/unified_attention_sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/unified_attention_sparse_mla.py)

Source documentation: This function computes the sparse attention. Note: topk_indices index the KV cache, not block_table.

Symbols: [`unified_attention_sparse_mla`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/unified_attention_sparse_mla.py#L6).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_SIZE`, `num_stages`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx942/triton/attention/sparse_attention_dsv4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/sparse_attention_dsv4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/sparse_mla_dsv4_train/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/sparse_mla_dsv4_train/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/sparse_mla_dsv4_train/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/sparse_mla_dsv4_train/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_pa_decode_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode_sparse.py)
- [op_tests/triton_tests/attention/test_pa_prefill_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill_sparse.py)
- [op_tests/triton_tests/attention/test_sparse_attention_dsv4_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_attention_dsv4_bwd.py)
- [op_tests/triton_tests/attention/test_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla.py)
- [op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py)
- [op_tests/test_jit_cache_transaction.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_jit_cache_transaction.py)
- [op_tests/test_pa_sparse_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_sparse_prefill.py)
- [op_tests/test_vsa_sparse_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_vsa_sparse_attention.py)

## Performance and validation

Include index preparation when comparing complete operators and validate empty selections, tail blocks and causal boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
