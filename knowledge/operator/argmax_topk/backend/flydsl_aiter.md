---
title: "Argmax topk: AITER flydsl"
kind: implementation_knowledge
operator: argmax_topk
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Argmax topk — AITER flydsl

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/herd_topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py) · **Recent**

Source documentation: HERD routing orchestration with FlyDSL candidate selection + finalize.

Symbols: [`_run_sigmoid_candidate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L22), [`_run_finalize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L56), [`_profile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L91), [`_unsupported_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L107), [`herd_topk_gating_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L152), [`herd_topk_gating`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/herd_topk.py#L171).

### `aiter/ops/flydsl/topk/topk_per_row.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py) · **Recent**

Source documentation: FlyDSL decode TopK interface.

Symbols: [`_validate_topk_signature`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L33), [`_validate_values_signature`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L96), [`_validate_flydsl_topk_call`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L115), [`_validate_radix_topk_one_block_call`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L160), [`_tensor_signature`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L190), [`_is_flydsl_topk_call_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L195), [`_is_flydsl_radix_topk_one_block_call_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L227), [`_is_flydsl_radix_topk_one_block_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L274), [`is_flydsl_top_k_per_row_decode_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L305), [`flydsl_top_k_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L330), [`flydsl_radix_topk_one_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row.py#L411).

### `aiter/ops/flydsl/topk/topk_per_row_argmax.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_argmax.py) · **Recent**

Source documentation: FlyDSL per-row argmax -- the k=1 selector.

Symbols: [`topk_per_row_argmax_serves`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_argmax.py#L38), [`topk_per_row_argmax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_argmax.py#L51).

### `aiter/ops/flydsl/topk/topk_per_row_small_k.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py) · **Recent**

Source documentation: FlyDSL per-row TopK for a small k (block-sparse indexer selection).

Symbols: [`_row_bucket`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L22), [`topk_per_row_small_k_serves`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L30), [`_unsupported_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L55), [`topk_per_row_small_k_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L82), [`_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L98), [`topk_per_row_small_k`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/topk/topk_per_row_small_k.py#L115).

### `aiter/ops/flydsl/kernels/herd_topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/herd_topk.py) · **Recent**

Source documentation: FlyDSL popularity and min-unique finalize kernel for HERD routing.

Symbols: [`build_herd_finalize_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/herd_topk.py#L19).

### `aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py) · **Recent**

Source documentation: Wave32/wave64 radix TopK with one block per input row. Short rows cache ordered keys in LDS. Long rows compact the selected radix bucket; stable modes preserve index ordering and tie-breaking. Rows with length <= k copy/pad in the same kernel without running radix.

Symbols: [`_build_bitonic_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py#L65), [`build_radix_topk_one_block_body`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py#L80), [`build_radix_topk_one_block_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py#L1395).

### `aiter/ops/flydsl/kernels/topk/topk_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py) · **Recent**

Source documentation: Scalar, load and wave helpers shared by the FlyDSL TopK kernels.

Symbols: [`_f32_to_ord`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py#L18), [`_row_length`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py#L26), [`_row_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py#L34), [`_load_f32x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py#L44), [`_warp_inclusive_prefix_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_common.py#L51).

### `aiter/ops/flydsl/kernels/topk/topk_per_row_argmax.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_argmax.py) · **Recent**

Source documentation: Per-row argmax: k=1, as a reduction rather than a selection. At k=1 the other selectors do work that has no answer to produce. The small-k one ranks chunks by their maximum to bound a survivor set -- but the top-1 chunk is just the chunk holding the maximum, and the survivors are whatever ties with it; the radix ones walk the key space for a threshold that is the answer itself.

Symbols: [`topk_per_row_argmax_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_argmax.py#L99), [`build_topk_per_row_argmax_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_argmax.py#L122).

### `aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py) · **Recent**

Source documentation: Single-launch variable-length decode TopK. Every CTA reads its row's length. A short row is one CTA writing the final output. A long row is split into parts: each part publishes a local TopK, counts in on a per-row counter, and the last part to arrive merges the candidates in the same launch. No CTA waits on another, so the grid never needs to be co-resident. Both the part and the merge run the one-block selector.

Symbols: [`_part_elements`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L45), [`topk_per_row_decode_parts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L51), [`build_topk_per_row_decode_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L66), [`_allocate_partials`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L281), [`_get_cached_partials`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L289), [`_get_cached_counter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L294), [`_get_workspace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L299), [`clear_topk_per_row_decode_workspace_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L310), [`launch_topk_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode.py#L316).

### `aiter/ops/flydsl/kernels/topk/topk_per_row_decode_persistent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode_persistent.py) · **Recent**

Source documentation: One-workgroup FlyDSL decode TopK.

Symbols: [`build_topk_per_row_decode_one_workgroup_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_decode_persistent.py#L39).

### `aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py) · **Recent**

Source documentation: Per-row TopK for a large k: read the row once, sort only what survives. A row of N fp32 scores, k in the hundreds to low thousands, N up to a million. The selectors this replaces walk the row once per radix pass and once more to scatter -- four trips to HBM for a row that is otherwise perfectly streamable. Here the row is read once.

Symbols: [`_prefetch_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L162), [`_lds_budgets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L179), [`_ugt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L214), [`_ord_unsigned`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L223), [`_wave_inclusive_prefix_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L247), [`topk_per_row_radix_stream_block_threads`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L258), [`_resolve_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L270), [`topk_per_row_radix_stream_lds_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L321), [`topk_per_row_radix_stream_serves`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L358), [`build_topk_per_row_radix_stream_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_radix_stream.py#L395).

### `aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py) · **Recent**

Source documentation: Per-row TopK for a small k, by narrowing to k chunks instead of sorting. The rows this serves are short and wide-ish: a block-sparse indexer scores one value per KV block, so even a 1M-token context is 8192 scores, and k is 16. Sorting every tile of such a row costs 13-20x a pass that only reads it. The narrowing: Partition the row into C >= k chunks and rank the chunks by their maximum, ties broken by chunk id.

Symbols: [`_uge`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py#L97), [`_ugt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py#L102), [`lds_limit`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py#L116), [`topk_per_row_small_k_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py#L131), [`build_topk_per_row_small_k_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/topk/topk_per_row_small_k.py#L154).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_topk.py)
- [op_tests/test_flydsl_topk_per_row_small_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_topk_per_row_small_k.py)
- [op_tests/test_topk_per_row.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row.py)
- [op_tests/test_topk_per_row_stable.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row_stable.py)
- [op_tests/test_topk_plain.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_plain.py)
- [op_tests/test_topk_row_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_row_prefill.py)
- [op_tests/test_topk_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_select.py)
- [op_tests/triton_tests/test_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_topk.py)
- [op_tests/triton_tests/torch_compile/test_compile_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_topk.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
