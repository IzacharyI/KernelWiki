---
title: "Residual addition and RMS normalization: AITER triton"
kind: implementation_knowledge
operator: fused_add_rmsnorm
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Residual addition and RMS normalization — AITER triton

The residual update and normalized output may both be observable. Fusing the reduction saves an intermediate memory pass, but aliasing and the order of rounding must follow the API.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py) · **Recent**

Source documentation: Fuses rmsnorm, add, and padding into a single kernel.

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py#L21), [`fused_add_rmsnorm_pad`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py#L29).

Parameters referenced in this file: `BLOCK_SIZE_N`, `num_warps`.

### `aiter/ops/triton/normalization/fused_rmsnorm_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py) · **Recent**

Source documentation: RMSNorm over the last dim of a 2D tensor, with an optional residual add. x: (M, N) tensor (bf16/fp16). Made contiguous if it isn't already. weight: (N,) tensor. res1: optional (M, N) residual; when given, computes x += res1 first and returns (out, out_res1) where out_res1 is the pre-norm sum. Returns out (M, N) if res1 is None, else (out, out_res1). Dispatches to a gfx1250 Gluon kernel when running on gfx1250, otherwise falls back to a portable Triton kernel.

Symbols: [`_fused_rmsnorm_add_core`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L18), [`_fused_rmsnorm_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L104), [`_fused_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L111), [`_fused_rmsnorm_add_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L118), [`_fused_rmsnorm_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L125), [`fused_rmsnorm_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add.py#L132).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/normalization/fused_rmsnorm_add_rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add_rmsnorm.py) · **Recent**

Source documentation: Fused RMSNorm, residual add, RMSNorm for Muse-style sandwich norms.

Symbols: [`_fused_rmsnorm_add_rmsnorm_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add_rmsnorm.py#L15), [`fused_rmsnorm_add_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/fused_rmsnorm_add_rmsnorm.py#L31).

Parameters referenced in this file: `BLOCK_SIZE_N`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/normalization/fused_add_rmsnorm_pad.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_add_rmsnorm_pad.py) · **Recent**

Symbols: [`_fused_add_rmsnorm_pad`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_add_rmsnorm_pad.py#L17).

Parameters referenced in this file: `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add.py) · **Recent**

Symbols: [`_triton_fused_rms_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add.py#L21).

Parameters referenced in this file: `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add_rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add_rmsnorm.py) · **Recent**

Source documentation: RMSNorm, residual add, and RMSNorm with FP32 intermediates.

Symbols: [`_fused_rmsnorm_add_rmsnorm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/fused_rmsnorm_add_rmsnorm.py#L18).

Parameters referenced in this file: `BLOCK_SIZE_N`.

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

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
