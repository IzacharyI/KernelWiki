---
title: "Causal one-dimensional convolution: AITER hip"
kind: implementation_knowledge
operator: causal_conv1d
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Causal one-dimensional convolution — AITER hip

Prefill and token-update paths differ in state movement, padding and Q/K/V splitting. The update path mutates a rolling history that must remain correct over subsequent tokens.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/kernels/causal_conv1d_fwd_split_qkv.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_fwd_split_qkv.cu) · **Recent**

Symbols: [`bf16_to_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_fwd_split_qkv.cu#L18), [`f32_to_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_fwd_split_qkv.cu#L27), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_fwd_split_qkv.cu#L46), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_fwd_split_qkv.cu#L215).

### `csrc/include/causal_conv1d_fwd_split_qkv.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/causal_conv1d_fwd_split_qkv.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/causal_conv1d_update.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/causal_conv1d_update.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/causal_conv1d_update.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_update.cu)

Symbols: [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/causal_conv1d_update.cu#L106).

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

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
