---
title: "Sparse attention: AITER opus"
kind: implementation_knowledge
operator: sparse_attention_nsa
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse attention — AITER opus

Sparse kernels consume selected pages, blocks or token indices. Selection granularity, duplicates, padding and causal masking affect the logical work independently of dense attention shapes.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/pa_sparse_prefill_opus.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py) · **Recent**

Source documentation: OPUS-based sparse paged prefill attention for DeepSeek-V4. Two-region sparse scaled-dot-product attention over a paged prefix source (unified_kv) and a flat per-fwd extend source (kv), with a per-head softmax-denominator sink. The two regions share a single online-softmax accumulator, making the order region-invariant. The user-facing entry is :func:pa_sparse_prefill_opus, which dispatches on the running GPU.

Symbols: [`_dispatch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L42), [`pa_sparse_prefill_gfx950_opus_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L58), [`pa_sparse_prefill_gfx950_opus_split_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L73), [`pa_sparse_prefill_gfx1250_opus_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L92), [`_auto_kv_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L113), [`_pa_sparse_prefill_opus_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L122), [`pa_sparse_prefill_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L139), [`pa_sparse_prefill_fp8_gfx950_opus_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L266), [`pa_sparse_prefill_fp8_gfx1250_opus_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L284), [`_pa_sparse_prefill_fp8_opus_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L301), [`pa_sparse_prefill_fp8_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pa_sparse_prefill_opus.py#L323).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4_bwd.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py)
- [op_tests/test_jit_cache_transaction.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_jit_cache_transaction.py)
- [op_tests/test_pa_sparse_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_sparse_prefill.py)
- [op_tests/test_vsa_sparse_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_vsa_sparse_attention.py)
- [op_tests/triton_tests/attention/test_pa_decode_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode_sparse.py)
- [op_tests/triton_tests/attention/test_pa_prefill_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill_sparse.py)
- [op_tests/triton_tests/attention/test_sparse_attention_dsv4_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_attention_dsv4_bwd.py)
- [op_tests/triton_tests/attention/test_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla.py)
- [op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py)

## Performance and validation

Include index preparation when comparing complete operators and validate empty selections, tail blocks and causal boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
