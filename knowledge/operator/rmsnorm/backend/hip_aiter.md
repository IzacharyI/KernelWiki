---
title: "RMS normalization: AITER hip"
kind: implementation_knowledge
operator: rmsnorm
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# RMS normalization — AITER hip

RMSNorm reduces squares across a row, forms an inverse RMS and scales by a learned weight. Row width, number of rows and vectorized access determine the balance between launch cost, parallelism and memory traffic.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/rmsnorm.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/rmsnorm.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/rmsnorm/rmsnorm_opus_norm.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_norm.hpp)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/rmsnorm/rmsnorm_opus_norm_bf16.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_norm_bf16.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp16.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp16.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp32.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_norm_fp32.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_rmsnorm.py)
- [op_tests/test_rmsnorm2d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rmsnorm2d.py)
- [op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py)
- [op_tests/triton_tests/normalization/test_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_rmsnorm.py)
- [op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py)

## Performance and validation

Check accumulation dtype, epsilon, weight dtype and strided/tail rows. Benchmark decode-size row counts separately from prefill.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
