---
title: "Causal one-dimensional convolution: AITER flydsl"
kind: implementation_knowledge
operator: causal_conv1d
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Causal one-dimensional convolution — AITER flydsl

Prefill and token-update paths differ in state movement, padding and Q/K/V splitting. The update path mutates a rolling history that must remain correct over subsequent tokens.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/causal_conv1d_flydsl.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py) · **Recent**

Source documentation: FlyDSL prefill causal-conv1d kernel with fused split q/k/v output.

Symbols: [`build_causal_conv1d_flydsl_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py#L20), [`_get_compiled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py#L382), [`_build_chunk_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py#L388), [`causal_conv1d_split_qkv_flydsl_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py#L406).

### `aiter/ops/flydsl/causal_conv1d_update_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py) · **Recent**

Source documentation: FlyDSL causal-conv1d update host wrappers (decode + speculative verify). One algorithm behind two upstream interfaces, both maintained: causal_conv1d_update_flydsl is a drop-in for vLLM's (and aiter's Triton) causal_conv1d_update; causal_conv1d_update_sglang_flydsl is SGLang's, which adds per-step intermediate_conv_window snapshots and an EAGLE tree traversal. Both launch the same grid shape, so one tile/occupancy policy (_pick_block_n / _pick_cpt) serves both.

Symbols: [`_raw_stream`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L46), [`_ceil_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L107), [`_env_int`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L111), [`_n_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L128), [`_target_wg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L149), [`_pick_cpt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L160), [`_pick_block_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L190), [`_is_supported_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L218), [`_dtype_out_of_scope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L226), [`_require_in_scope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L259), [`_shapes_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L289), [`_index_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L324), [`_require_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L333), [`_causal_conv1d_update_flydsl_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L341), [`_causal_conv1d_update_sglang_flydsl_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L381), [`_resolve_activation`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L429), [`causal_conv1d_update_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L440), [`_is_dedup_conv_window`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L661), [`causal_conv1d_update_sglang_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_update_kernels.py#L676).

Parameters referenced in this file: `BLOCK_N`.

### `aiter/ops/flydsl/kernels/causal_conv1d_update.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py) · **Recent**

Source documentation: FlyDSL decode + speculative-verify causal-conv1d update kernels. Two builders, one per upstream interface, sharing a module because they are the same convolution behind two shells: * build_causal_conv1d_update_module ports vLLM's Triton _causal_conv1d_update_kernel, i.e. the *chain* (linear) speculative path. * build_causal_conv1d_update_sglang_module ports SGLang's kernel of the same name.

Symbols: [`_vec_chunks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L41), [`_rsrc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L54), [`_addr_at`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L59), [`_make_buffer_io`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L65), [`build_causal_conv1d_update_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L124), [`compile_causal_conv1d_update`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L535), [`build_causal_conv1d_update_sglang_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L570), [`compile_causal_conv1d_update_sglang`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/causal_conv1d_update.py#L1052).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_causal_conv1d_prefill_split_qkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_prefill_split_qkv.py)
- [op_tests/test_causal_conv1d_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_update.py)
- [op_tests/test_flydsl_causal_conv1d_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_causal_conv1d_update.py)
- [op_tests/triton_tests/conv/test_causal_conv1d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_causal_conv1d.py)
- [op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py)
- [op_tests/triton_tests/utils/causal_conv1d_update_refs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/causal_conv1d_update_refs.py)

## Performance and validation

Test a sequence against the full convolution reference. Include state initialization, optional activation and variable sequence boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
