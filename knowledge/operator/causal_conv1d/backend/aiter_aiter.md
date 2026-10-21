---
title: "Causal one-dimensional convolution: AITER aiter"
kind: implementation_knowledge
operator: causal_conv1d
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Causal one-dimensional convolution — AITER aiter

Prefill and token-update paths differ in state movement, padding and Q/K/V splitting. The update path mutates a rolling history that must remain correct over subsequent tokens.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/causal_conv1d_fwd_split_qkv.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py) · **Recent**

Source documentation: Prefill causal-conv1d (HIP) with fused split q/k/v output. Drop-in replacement for the Triton causal_conv1d_split_qkv_triton_fn. x: [dim, cu_seqlen] bf16, channels are concatenated [Q \| K \| V]. weight: [dim, width]; conv_states: [num_cache_lines, dim, width-1] (in/out). Returns contiguous (q, k, v) of shapes [cu_seqlen, k_dim] / [cu_seqlen, v_dim].

Symbols: [`causal_conv1d_fwd_split_qkv_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py#L22), [`_pick_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py#L53), [`_build_chunk_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py#L60), [`causal_conv1d_split_qkv_hip_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py#L90).

### `aiter/ops/causal_conv1d_update.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_update.py) · **Recent**

Source documentation: Causal 1D convolution update with state management (for inference/decoding). This function is designed for autoregressive generation where we process one (or a few) new tokens at a time and maintain a sliding window state buffer.

Symbols: [`causal_conv1d_update`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_update.py#L12).

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

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
