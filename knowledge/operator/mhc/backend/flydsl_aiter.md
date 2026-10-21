---
title: "Manifold-constrained hyper-connections: AITER flydsl"
kind: implementation_knowledge
operator: mhc
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Manifold-constrained hyper-connections — AITER flydsl

MHC paths implement pre/post residual mixing and normalization-related stages, including Sinkhorn-style transforms in selected entries. Fusion changes intermediate storage and reduction order.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/mega_mhc_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py) · **Recent**

Source documentation: Host wrapper for the single-launch FlyDSL Mega-mHC seam (DeepSeek-V4.1, gfx950). Same arguments and returns as the Triton seam aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm plus out_dtype ("bf16", "fp8_grid" or "mxfp8") and a config knob override. See aiter/ops/flydsl/kernels/mega_mhc.py for the kernel.

Symbols: [`_seam_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L78), [`_with_seg128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L84), [`_check_out_dtype`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L100), [`_require`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L108), [`get_mega_mhc_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L114), [`_with_dist_finish`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L197), [`_Scratch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L228), [`_prepack_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L271), [`_dist_guard_error`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L309), [`_check_dist_guard`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L321), [`_cu_num`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L329), [`_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L333), [`flydsl_mega_mhc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_mhc_kernels.py#L337).

Parameters referenced in this file: `BLOCK_M`, `NUM_KSPLIT`.

### `aiter/ops/flydsl/kernels/mega_mhc.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py) · **Recent**

Source documentation: Single-launch Mega-mHC seam for DeepSeek-V4.1 (delayed / Single-Pass mHC), gfx950.

Symbols: [`_NS`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L151), [`_put`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L155), [`_ld`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L164), [`_st`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L170), [`ue8m0_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L189), [`smem_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L237), [`max_x1_lds_slots`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L256), [`check_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L270), [`dist_residency_error`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L352), [`kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L378), [`grid_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L405), [`compile_mega_mhc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_mhc.py#L415).

Parameters referenced in this file: `BLOCK_M`, `NUM_KSPLIT`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mhc_res_layout.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mhc_res_layout.py) · **Recent**

Source documentation: mHC residual layout conversions (FlyDSL). The mHC gemm reads and writes the residual in a shuffled layout:: resS[kb][head][row][kk] <- res[row][head][k], k = kb * KS + kk KS is the kernel-side tile width, so one kk run is a contiguous KS * itemsize byte block in *both* layouts.

Symbols: [`units_per_run`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mhc_res_layout.py#L57), [`to_shuffled_row_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mhc_res_layout.py#L70), [`build_mhc_res_layout_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mhc_res_layout.py#L76).

Parameters referenced in this file: `BLOCK_THREADS`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py)
- [op_tests/op_benchmarks/triton/bench_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc.py)
- [op_tests/op_benchmarks/triton/bench_mhc_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/test_flydsl_mega_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mega_mhc.py)
- [op_tests/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mhc.py)
- [op_tests/triton_tests/fusions/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc.py)
- [op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/triton_tests/utils/mhc_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mhc_ref.py)

## Performance and validation

Match channel count, mixing matrices, iteration settings and all residual outputs. Include surrounding pre/post work in a fused comparison.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
