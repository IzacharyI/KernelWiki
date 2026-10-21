---
title: "Residual addition and RMS normalization: AITER gluon"
kind: implementation_knowledge
operator: fused_add_rmsnorm
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Residual addition and RMS normalization — AITER gluon

The residual update and normalized output may both be observable. Fusing the reduction saves an intermediate memory pass, but aliasing and the order of rounding must follow the API.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_add_rmsnorm_pad.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_add_rmsnorm_pad.py) · **Recent**

Symbols: [`_rmsmorm_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_add_rmsnorm_pad.py#L13), [`_gluon_fused_add_rmsnorm_pad_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_add_rmsnorm_pad.py#L29).

Parameters referenced in this file: `BLOCK_SIZE_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_rmsnorm_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_rmsnorm_add.py) · **Recent**

Symbols: [`_rmsnorm_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_rmsnorm_add.py#L10), [`_gluon_fused_rms_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/normalization/fused_rmsnorm_add.py#L18).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/normalization/fused_add_rmsnorm_pad/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/normalization/fused_add_rmsnorm_pad/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_add_rmsnorm_pad.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_add_rmsnorm_pad.py)
- [op_tests/op_benchmarks/triton/bench_fused_rmsnorm_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_rmsnorm_add.py)
- [op_tests/op_benchmarks/triton/bench_fused_rmsnorm_add_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_rmsnorm_add_rmsnorm.py)
- [op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py)
- [op_tests/triton_tests/normalization/test_fused_rmsnorm_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_fused_rmsnorm_add.py)
- [op_tests/triton_tests/normalization/test_fused_rmsnorm_add_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_fused_rmsnorm_add_rmsnorm.py)

## Performance and validation

Validate every output and in-place buffer, including optional padding. Compare the full unfused sequence against the fused wrapper.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
