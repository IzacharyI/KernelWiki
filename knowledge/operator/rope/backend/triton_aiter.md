---
title: "Rope: AITER triton"
kind: implementation_knowledge
operator: rope
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Rope — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/rope/rope.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py) · **Recent**

Symbols: [`RotateStyle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L43), [`_rope_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L49), [`rope_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L107), [`rope_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L132), [`_rope_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L156), [`rope_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L214), [`_rope_thd_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L239), [`rope_thd_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L299), [`rope_thd_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L326), [`_rope_thd_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L352), [`rope_thd_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L412), [`_rope_cached_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L440), [`rope_cached_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L508), [`rope_cached_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L537), [`rope_cached_positions_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L565), [`rope_cached_positions_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L595), [`rope_cached_positions_offsets_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L624), [`rope_cached_positions_offsets_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L655), [`_rope_cached_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L685), [`rope_cached_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L793), [`rope_cached_positions_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L822), [`rope_cached_positions_offsets_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L852), [`_rope_cached_thd_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L883), [`rope_cached_thd_positions_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1065), [`rope_cached_thd_positions_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1098), [`rope_cached_thd_positions_offsets_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1131), [`rope_cached_thd_positions_offsets_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1165), [`_rope_cached_thd_positions_offsets_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1199), [`rope_cached_thd_positions_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1381), [`rope_cached_thd_positions_offsets_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1414), [`_rope_fwd_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1448), [`rope_fwd_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1485), [`rope_fwd_2d_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1524), [`rope_fwd_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1556), [`RoPE`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1616), [`RoPETHD`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1653), [`RoPECached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1689), [`RoPE2D`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/rope.py#L1735).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_L`, `BLOCK_N`, `BLOCK_S`, `BLOCK_T`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/rope/rope.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py) · **Recent**

Symbols: [`_get_neox_rotated_x_1D`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L11), [`_get_gptj_rotated_x_1D`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L26), [`_get_neox_rotated_x`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L40), [`_get_gptj_rotated_x`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L67), [`_rope_kernel_sbhd_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L108), [`_rope_kernel_sbhd_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L227), [`_rope_kernel_thd_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L351), [`_rope_kernel_thd_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L475), [`_rope_kernel_sbhd_cached_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L601), [`_rope_kernel_sbhd_cached_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L741), [`_rope_kernel_thd_cached_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L883), [`_rope_kernel_thd_cached_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L1079), [`_rope_kernel_cached_thd_2c_gqa_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L1276), [`_rope_kernel_cached_thd_2c_gqa_onehead_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L1474), [`_rope_kernel_cached_thd_2c_gqa_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L1668), [`_rope_kernel_cached_thd_2c_gqa_onehead_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L1866), [`_rope_fwd_2d_kernel_neox`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L2052), [`_rope_fwd_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/rope.py#L2181).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_L`, `BLOCK_N`, `BLOCK_S`, `BLOCK_T`, `num_stages`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_qk_rope_cat/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_qk_rope_cat/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_qk_rope_reshape_and_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_rope_reshape_and_cache.py)
- [op_tests/op_benchmarks/triton/bench_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_rope.py)
- [op_tests/triton_tests/rope/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/__init__.py)
- [op_tests/triton_tests/rope/test_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_rope.py)
- [op_tests/triton_tests/rope/test_rope_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_rope_mxfp4.py)
- [op_tests/triton_tests/torch_compile/test_compile_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_rope.py)
- [op_tests/triton_tests/utils/rotary_embedding.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/rotary_embedding.py)
- [op_tests/test_indexer_qk_rope_quant_and_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_indexer_qk_rope_quant_and_cache.py)
- [op_tests/test_inverse_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_inverse_rope_group_quant.py)
- [op_tests/test_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rope.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
