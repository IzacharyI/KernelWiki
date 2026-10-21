---
title: "RMS normalization: AITER opus"
kind: implementation_knowledge
operator: rmsnorm
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# RMS normalization — AITER opus

RMSNorm reduces squares across a row, forms an inverse RMS and scales by a learned weight. Row width, number of rows and vectorized access determine the balance between launch cost, parallelism and memory traffic.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/opus/rmsnorm_opus_kernel.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp) · **Recent**

Symbols: [`rmsnorm_opus_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L35), [`rmsnorm_quant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L50), [`rmsnorm_opus_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L68), [`rmsnorm_quant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L72), [`quant_cast`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L79), [`block_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L89), [`rmsnorm_opus_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L117), [`rmsnorm_quant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/rmsnorm_opus_kernel.hpp#L244).

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

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
