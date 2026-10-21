---
title: "Paged and unified attention decode: AITER gluon"
kind: implementation_knowledge
operator: attention_decode_paged
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Paged and unified attention decode — AITER gluon

Paged attention must preserve the block table, sequence lengths, causal mask, query-head to KV-head mapping and cache dtype. Split-KV exposes parallelism at the cost of partial buffers and a reduction pass.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/gluon/pa_decode_gluon.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py) · **Recent**

Source documentation: Get CDNA version lazily to avoid CUDA initialization during import.

Symbols: [`_load_kv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L61), [`_fused_max_combine`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L68), [`get_cdna_version`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L73), [`get_occupancy`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L83), [`get_recommended_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L87), [`define_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L120), [`store_temporary_result`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L187), [`paged_attention_decode_v2_gluon_large_block_dot_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L245), [`paged_attention_decode_sliding_window_head_1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L1038), [`paged_attention_decode_sliding_window`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L2240), [`paged_attention_decode_v2_gluon_dot_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L3254), [`paged_attention_decode_ps_reduce_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L4066), [`paged_attention_decode_v2_reduce_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L4169), [`_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L4406), [`_paged_attention_decode_v2_reduce_kernel_wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L4639), [`pa_decode_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_decode_gluon.py#L4750).

Parameters referenced in this file: `num_stages`, `waves_per_eu`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py) · **Recent**

Source documentation: Reduce-max that propagates NaN. Skipping NaN handling is extra work on AMD.

Symbols: [`elementwise_max_prop_nan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L18), [`reduce_max_prop_nan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L23), [`AttentionConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L30), [`TDMKVLoaderConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L238), [`TDMKVLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L286), [`TDMGatherKVLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L503), [`AttentionProgram`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L697), [`attention_loop_standard`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1182), [`attention_loop_reordered`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1262), [`attention_loop_tensor_subtile_split`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1449), [`load_q`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1754), [`find_seq_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1803), [`_unified_attention_gluon_kernel_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py#L1844).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SIZE`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py) · **Recent**

Source documentation: Configuration for unified attention layouts and derived constants.

Symbols: [`apply_softcap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L31), [`AttentionConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L40), [`AttentionProgram`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L562), [`find_seq_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1789), [`get_q_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1811), [`cdiv_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1830), [`get_seq_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1835), [`e2m1_packed_to_fp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1852), [`_unified_attention_gluon_kernel_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py#L1927).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SCALES_SIZE`, `BLOCK_SIZE`, `BLOCK_SIZE_INNER_DIM_SHFL`, `BLOCK_SIZE_N_SHFL`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py) · **Recent**

Source documentation: Gluon (gfx950) paged decode attention with a sigmoid output gate and an optional group-128 FP8 epilogue. Targets Qwen3-Next-style full-attention layers. Split-K (flash-decoding) over the context: _attention_partials writes one (numerator, denominator, maximum) record per split, then _merge_gate_quantize reduces them, applies out * sigmoid(gate), and optionally emits group-128 FP8.

Symbols: [`_attention_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py#L46), [`_attention_partials`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py#L162), [`_merge_prefix`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py#L335), [`_merge_gate_quantize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate.py#L384).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py) · **Recent**

Source documentation: Short-context Gluon (gfx950) body for paged decode attention with a sigmoid output gate and an optional group-128 FP8 epilogue. Same op, same contract and same outputs as the long-context body in paged_attention_output_gate.py; a different decomposition of the same work.

Symbols: [`_cache_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L74), [`_load_page_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L93), [`_prefetch_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L108), [`_sigmoid_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L156), [`_quant_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L162), [`_split_record`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L175), [`_head_softmax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L190), [`_short_attention_partials`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L237), [`_finish_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L566), [`_short_finish`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/paged_attention_output_gate_short_context.py#L719).

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py) · **Recent**

Source documentation: Reduce-max that propagates NaN. Skipping NaN handling is extra work on AMD.

Symbols: [`_async_copy_accepts_distributed_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L34), [`elementwise_max_prop_nan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L58), [`reduce_max_prop_nan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L63), [`_offset_bases_to_blocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L69), [`_padded_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L104), [`_shuffled_kv_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L129), [`_padded_kv_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L170), [`_legacy_kv_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L222), [`_make_cdna4_kv_load_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L246), [`AttentionConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L279), [`AsyncKVLoaderConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L471), [`AsyncKVLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L503), [`AsyncGatherKVLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L766), [`AttentionProgram`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L1082), [`attention_loop_single_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L1470), [`attention_loop_standard`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L1546), [`find_seq_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L1684), [`_unified_attention_gluon_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py#L1705).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SIZE`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_reduce.py)

Source documentation: Gluon split-KV reduce for gfx1250 decode attention (replaces the Triton reduce_segments). One workgroup per query token merges the per-segment partials with a log-sum-exp reduction; heads are split across waves and the segment axis is held in-thread, so the merge is pure in-register with no cross-wave communication. Gated to all-decode, HEAD_SIZE_PADDED % 32 == 0, num_query_heads % NUM_WARPS == 0, and small NUM_SEGMENTS (else the wrapper falls back to Triton).

Symbols: [`reduce_segments_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_reduce.py#L18).

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/unified_attention/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/cpp/test_pa_decode_gluon_aot.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/test_pa_decode_gluon_aot.py)
- [op_tests/op_benchmarks/triton/bench_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_decode.py)
- [op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py)
- [op_tests/op_benchmarks/triton/bench_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_unified_attention.py)
- [op_tests/test_flydsl_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_decode.py)
- [op_tests/test_pa_decode_bf16_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_decode_bf16_asm.py)
- [op_tests/triton_tests/attention/test_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode.py)
- [op_tests/triton_tests/attention/test_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_paged_attention_output_gate.py)
- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention.py)
- [op_tests/triton_tests/test_pa_decode_gluon.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_pa_decode_gluon.py)

## Performance and validation

Record page size, KV length distribution, head dimensions, number of query tokens and split count. Time the full split/reduce path; verify multi-token decode causality.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
