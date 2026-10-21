---
title: "Three-dimensional convolution: AITER flydsl"
kind: implementation_knowledge
operator: conv3d
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Three-dimensional convolution — AITER flydsl

AITER contains implicit-GEMM, im2col, transpose and specialized 3D convolution paths. The cost of input preparation can dominate a fast inner GEMM.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/conv3d_policy.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py) · **Recent**

Source documentation: Shape-aware launch-config enumeration for FlyDSL conv3d. Closed-form legality via validate_launch_config. Includes 48/96/192 N tiles so VAE Cout 96/192/384 can fill TILE_N, unlike the runtime power-of-two ladder.

Symbols: [`is_legal_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L78), [`tile_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L89), [`lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L93), [`max_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L97), [`_ceil_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L108), [`_n_fill`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L112), [`_sweep`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L121), [`get_flydsl_conv3d_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/conv3d_policy.py#L198).

Parameters referenced in this file: `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py) · **Recent**

Source documentation: Shared pieces of the conv3d kernels (geometry, grid, scatter). Gather is in conv3d_im2col.

Symbols: [`_as_stream`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L38), [`buffer_atomic_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L42), [`barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L47), [`sgpr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L53), [`flat_buffer_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L58), [`in_range`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L66), [`magic_u32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L74), [`Divisor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L98), [`static_divisor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L124), [`pack_reciprocal`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L134), [`dyn_divisor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L147), [`dil`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L157), [`gather_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L162), [`validate_launch_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L190), [`TileConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L212), [`make_tile_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L246), [`weight_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L279), [`WeightLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L288), [`b_park_elems`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L343), [`make_shared_storage`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L348), [`LdsStager`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L385), [`MmaTiling`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L428), [`ConvGeometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L569), [`out_extent`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L582), [`make_conv_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L587), [`ConvExtents`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L610), [`StaticInputExtents`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L664), [`static_input_extents`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L674), [`static_extents`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L686), [`DynShapeArgs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L714), [`dyn_shape_values`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L751), [`unit_divisors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L782), [`dyn_extents`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L792), [`LaunchGrid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L829), [`make_launch_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L867), [`BlockCoords`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L942), [`_xcd_contiguous`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L957), [`block_coords`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L976), [`OutputScatterPlan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L1055), [`make_output_scatter_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L1084), [`OutputScatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_gfx950_utils.py#L1147).

Parameters referenced in this file: `BLOCK_THREADS`, `BLOCK_VECS`, `splitK`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/conv/conv3d_im2col.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py) · **Recent**

Source documentation: Im2col gather for implicit-GEMM conv3d: tap addressing, padding, buffer rebasing.

Symbols: [`_bytes_of`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L33), [`Im2colPlan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L44), [`make_im2col_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L108), [`strip_elems`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L269), [`_base_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L274), [`_reuse_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L278), [`patch_needs_park`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L284), [`_patch_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L289), [`plan_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L297), [`StripGather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L306), [`PatchGather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L394), [`Im2colGather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_im2col.py#L479).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py) · **Recent**

Source documentation: Double-buffered implicit-GEMM conv3d (BF16). Dispatch lives in conv_kernels.py; pre-transpose in conv3d_transpose.py.

Symbols: [`Conv3dImplicitParam`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L53), [`make_conv3d_implicit_param`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L91), [`_dyn_hw_closure_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L154), [`_shape_agnostic_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L164), [`_assert_shape_agnostic`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L179), [`compile_conv3d_implicit`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/conv/conv3d_implicit_gfx950.py#L245).

Parameters referenced in this file: `BLOCK_THREADS`, `tile_k`, `tile_m`, `tile_n`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_conv3d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_conv3d.py)
- [op_tests/triton_tests/conv/test_conv3d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_conv3d.py)

## Performance and validation

Record tensor layout, kernel size, stride, padding and dilation. Include transforms/im2col in the full operator timing.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
