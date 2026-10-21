---
title: "Layout shuffle: AITER flydsl"
kind: implementation_knowledge
operator: layout_shuffle
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Layout shuffle — AITER flydsl

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/mxscale_preshuffle_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py) · **Recent**

Source documentation: Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA). Operand convention (the weight is preshuffled caller-side): A : [M, K] row-major, NOT preshuffled (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte) B : preshuffled via aiter.ops.shuffle.shuffle_weight(., (16, 16)) (fp4 or fp8 weight) a_scale / b_scale : E8M0, each a block and a layout (flydsl_mxscale_preshuffle_gemm).

Symbols: [`_gemm_exe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L40), [`_reduce_exe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L50), [`_fused_splitk_buffers`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L59), [`flydsl_mxscale_preshuffle_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L76), [`_warn_untuned`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L371), [`_heuristic_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L383), [`run_gemm_a8w8_mxfp8_gfx950`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L424), [`pick_mxfp8_kernel_name_gfx950`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxscale_preshuffle_kernels.py#L509).

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/layout_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py) · **Recent**

Source documentation: Layout helpers for GEMM kernels. Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes idx2crd / crd2idx with plain arith ops for static layouts. Falls back to fly dialect ops for dynamic layouts. Optimisation: power-of-2 strides/shapes emit shrui / andi instead of divui / remui, avoiding 10-15-cycle V_DIV sequences on CDNA GPUs.

Symbols: [`_is_pow2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L18), [`_div_pow2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L23), [`_mod_pow2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L33), [`_parse_dim`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L41), [`_parse_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L47), [`_has_dynamic_strides`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L58), [`idx2crd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L63), [`crd2idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L111), [`get`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/layout_utils.py#L144).

### `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py) · **Recent**

Source documentation: Shared MFMA preshuffle helpers for preshuffle GEMM kernels. Key primitives: - B preshuffle layout builder (supports byte-packed element types, incl. packed int4) - B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)

Symbols: [`crd2idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L19), [`swizzle_xor16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L24), [`split_row_major_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L36), [`_buffer_load_vec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L41), [`PreshuffleScaleLayout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L79), [`make_preshuffle_scale_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L94), [`PreshuffleBLayout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L143), [`make_preshuffle_b_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L150), [`tile_chunk_coord_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L211), [`buffer_copy_gmem16_dwordx4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L230), [`_lds_store_xor16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L253), [`lds_store_16b_xor16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L276), [`lds_store_8b_xor16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L302), [`lds_store_4b_xor16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L328), [`xcd_remap_bx_by`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py#L354).

Parameters referenced in this file: `tile_m`, `tile_n`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_qk_concat.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_concat.py)
- [op_tests/triton_tests/fusions/test_fused_qk_concat.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_qk_concat.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
