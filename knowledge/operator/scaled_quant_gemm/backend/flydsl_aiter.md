---
title: "Scaled and mixed-precision GEMM: AITER flydsl"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER flydsl

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py) · **Recent**

Source documentation: High-level gfx950 dense BF16 x MXFP4 FlyDSL API.

Symbols: [`DenseGemmConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L35), [`PreshuffledA16WFP4Weight`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L46), [`_require_gfx950`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L55), [`a16wfp4_shape_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L61), [`a16wfp4_config_legal`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L66), [`_select_tile_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L97), [`_select_k_wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L107), [`_decode_tile_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L114), [`_decode_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L130), [`_select_gemm_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L179), [`_validate_buffer_limits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L205), [`prepare_gemm_a16wfp4_weight`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L216), [`flydsl_gemm_a16wfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py#L264).

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py) · **Recent**

Source documentation: 8-wave CDNA4 (gfx950) backend for the FlyDSL a8w8 ptpc bpreshuffle GEMM.

Symbols: [`_lazy_import`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L31), [`lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L45), [`_validate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L50), [`compile_8wave_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L109), [`_as_1d_fp32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L131), [`_as_i8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L149), [`flydsl_8wave_gemm_a8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L154), [`parse_8wave_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L251), [`run_gemm_a8w8_bpreshuffle_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py#L269).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a8w8_mxfp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py) · **Recent**

Source documentation: FlyDSL mxfp8 GEMM on a (16, 16)-preshuffled weight, on whichever arch has a kernel for the operands' w_scale block and scale layouts -- the 2-D counterpart of batched_gemm_a8w8. XQ [M, K] fp8, WQ [N, K] fp8 preshuffled, x_scale / w_scale e8m0, Out [M, N] bf16 / fp16. Each scale's layout is read off its rank: a 2-D scale is the unshuffled row-major one ("row": x_scale [M, K / 32], w_scale [N / WN, K / WK]), a 1-D one a pre-shuffled flat buffer ("mfma_tile").

Symbols: [`_runner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py#L38), [`_scale_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py#L49), [`gemm_a8w8_mxfp8_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py#L53), [`run_gemm_a8w8_mxfp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py#L61).

### `aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py) · **Recent**

Source documentation: 8-wave CDNA4 (gfx950) backend for the FlyDSL mxfp8 (1x32 e8m0) bpreshuffle GEMM.

Symbols: [`_lazy_import`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L35), [`lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L49), [`_validate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L54), [`compile_mxfp8_8wave_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L104), [`_as_u8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L128), [`flydsl_mxfp8_8wave_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L133), [`make_mxfp8_8wave_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L274), [`parse_mxfp8_8wave_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L281), [`run_gemm_a8w8_mxfp8_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py#L299).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py) · **Recent**

Source documentation: Tune spaces for every FlyDSL a8w8 (ptpc) bpreshuffle pipeline on CDNA. One operator, two pipelines, swept together under the same flydsl libtype so a single --libtype flydsl run picks one winner per shape: * preshuffle -- 4-wave MFMA, gfx942 + gfx950, fp8 or int8 weights. * 8wave -- 8-wave CDNA4 MFMA_Scale, gfx950 only, fp8 only.

Symbols: [`kernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L38), [`_ki`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L87), [`_smem_align`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L117), [`_smem_finalize_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L123), [`preshuffle_gemm_estimated_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L131), [`kernel_instance_estimated_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L156), [`max_lds_bytes_for_tune`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L169), [`_padded_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L179), [`kernel_fits_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L195), [`_vgpr_per_simd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L296), [`_estimate_max_wpe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L308), [`k_split_candidates`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L345), [`_build_kernels_list`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L372), [`EightWaveKernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L464), [`kernel_instance_estimated_lds_bytes_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L481), [`kernel_fits_shape_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L486), [`is_8wave_enabled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L509), [`_build_kernels_list_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L520), [`Pipeline`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py#L544).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_K_8WAVE`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py) · **Recent**

Source documentation: Full-style gfx1250 WMMA candidates for a8w8 (ptpc) bpreshuffle GEMM.

Symbols: [`WmmaKernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L38), [`_align_up`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L58), [`_ceil_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L62), [`_tile_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L66), [`_cluster_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L80), [`kernel_instance_estimated_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L84), [`_build_kernels_list`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L100), [`kernel_fits_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py#L127).

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py) · **Recent**

Source documentation: Full-style gfx1250 WMMA candidates for mxfp8_128 bpreshuffle GEMM.

Symbols: [`WmmaKernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L64), [`_align_up`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L92), [`_ceil_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L96), [`_tile_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L100), [`_cluster_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L126), [`_mxfp8_128_stage_scale_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L130), [`kernel_instance_estimated_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L138), [`_build_kernels_list`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L162), [`kernel_fits_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L207), [`compute_grid_ctas`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L288), [`is_compute_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py#L294).

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py) · **Recent**

Source documentation: Tune catalog for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA). Mirrors the aiter a8w8-bpreshuffle tune catalog style (kernelInstance + name + build_kernels_list), but the mxscale_preshuffle kernel exposes only tile_m/n/k + waves_per_eu as perf knobs (no async/lds/cshuffle/xcd/scheduler/split-K), so the candidate schema and kernelName grammar are correspondingly small.

Symbols: [`_estimate_max_wpe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L63), [`_a_row_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L83), [`make_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L88), [`kernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L124), [`parse_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L158), [`row_scale_panel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L198), [`col_scale_panel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L214), [`estimated_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L223), [`_max_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L245), [`instance_valid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L254), [`fits_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L284), [`_build_kernels_list`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L354), [`candidates_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L390), [`EightWaveKernelInstance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L425), [`fits_shape_8wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L435), [`Pipeline`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L490), [`_fits_mxpsh_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L501), [`_fits_mxpsh_row_w32x32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L507), [`_fits_8wave_w32x32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L513), [`_fits_mxpsh_row_w128x128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L517), [`_fits_8wave_w128x128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py#L523).

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/mxfp4_gemm1_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm1_kernels.py) · **Recent**

Source documentation: Launch GEMM1; v2 output keeps payload rows in expert-sorted order.

Symbols: [`_get_compiled_mxfp4_gemm1_port`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm1_kernels.py#L14), [`_assert_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm1_kernels.py#L64), [`flydsl_mxfp4_gemm1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm1_kernels.py#L187).

### `aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py) · **Recent**

Source documentation: gfx1250 FlyDSL backend for mxfp8_128 / mxfp8_32 bpreshuffle GEMM. mxfp8_128: fp8 e4m3 activations/weights with fp8_e8m0 block scales over a 128-element block (distinct from the fp32-scale blockscale GEMM). mxfp8_32: A scales are row-major (M, K//32) by default, optionally 32x4 shuffled; B scales are always 32x4 shuffled (N, K//32).

Symbols: [`_fused_splitk_ok`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L40), [`_lazy_import`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L50), [`_require_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L75), [`check_persistent_n_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L91), [`_run_mxfp8_bpreshuffle_gemm_a8_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L124), [`_parse_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L460), [`parse_wmma_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L476), [`parse_mxfp8_32_wmma_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L483), [`compute_kernel_k_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L491), [`cluster_m_grid_ok`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L496), [`resolve_cluster_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L504), [`cluster_m_fallback_values`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L521), [`is_compute_wmma_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L531), [`_resolved_wmma_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L540), [`run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L565), [`run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py#L614).

Parameters referenced in this file: `BLOCK_K`, `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py) · **Recent**

Source documentation: MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

Symbols: [`_load_partial`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L46), [`_make_dma_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L56), [`_scale_mma_atoms`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L84), [`_bq_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L99), [`_launch_gemm_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L119), [`launch_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L1557), [`launch_gemm_fused`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L1631), [`launch_splitk_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py#L1716).

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4.py) · **Recent**

Source documentation: gfx950 dense BF16 x MXFP4 GEMM kernel. The activation is consumed as BF16 without quantization. Preshuffled packed E2M1 weights and E8M0 block scales are loaded directly into registers, where v_cvt_scalef32_pk_bf16_fp4 produces transient BF16 fragments for mfma_f32_16x16x32_bf16. A dequantized weight matrix is never written to global memory. The K loop double-buffers A-LDS. Prefill (k_wave=1) still issues the next A/B tile before MFMA.

Symbols: [`compile_gemm_a16wfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4.py#L43).

Parameters referenced in this file: `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py) · **Recent**

Source documentation: Components for the gfx950 dense BF16 x MXFP4 GEMM.

Symbols: [`DenseGemmTraits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L20), [`make_dense_operand_loaders`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L122), [`make_dense_accumulators`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L192), [`load_b_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L205), [`advance_k_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L217), [`load_a_frags`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L224), [`_mma_order`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L232), [`compute_dense_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L245), [`reduce_dense_k_wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L293), [`make_dense_store`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L325), [`store_dense_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py#L349).

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py) · **Recent**

Source documentation: A8W8 (FP8 E4M3 activation x FP8 E4M3 weight) 256x256 / 128x256 MXscale GEMM for gfx1250.

Symbols: [`launch_gemm_a8w8_256x256`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py#L35).

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py) · **Recent**

Source documentation: 8-wave FP8 matmul with row-wise scaling for AMD CDNA4 (gfx950 only).

Symbols: [`make_fp8_buffer_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L17), [`swizzle_128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L33), [`compute_global_swizzle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L40), [`pack_i32x4_i32x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L62), [`wait_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L67), [`G2SLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L77), [`S2RLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L106), [`StoreC`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L141), [`Mfma16x16x128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L223), [`_xcd_swizzle_any`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L287), [`compile_fp8_gemm_8w`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py#L311).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py) · **Recent**

Source documentation: FP8xFP8 preshuffle GEMM kernels for gfx1250.

Symbols: [`launch_gemm_a8w8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py#L35).

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mxfp4_gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py) · **Recent**

Source documentation: Pick the deepest K pipeline that still fits two workgroups per CU.

Symbols: [`gemm1_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L47), [`_gemm1_body`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L58), [`_bm_constants`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L1460), [`default_k_stages`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L1478), [`default_epi_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L1491), [`compile_gemm1_a4w4_port`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L1497).

### `aiter/ops/flydsl/kernels/mxfp4_gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py) · **Recent**

Symbols: [`aq_bytes_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L45), [`saq_slot_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L49), [`tiling`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L53), [`_issue_a_load_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L59), [`compile_gemm2_a4w4_port`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L78), [`_gemm2_body`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L306), [`_flat_bf16_epilog`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L621), [`_cshuffle_flat_bf16_epilog`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L646), [`_flat_mxfp4_epilog`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L695), [`_atomic_bf16_epilog`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py#L802).

### `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py) · **Recent**

Source documentation: XOR16 swizzle for an FP8 LDS row whose width is 128 or 256 bytes.

Symbols: [`_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L20), [`_udiv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L26), [`_umod`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L30), [`_scale_mma_atoms`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L37), [`_lds_ptr3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L57), [`_gep3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L62), [`_global_base_ptr1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L68), [`_gep1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L73), [`_global_ptr1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L79), [`_global_i32_at`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L83), [`_global_f32_at`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L87), [`_global_i32_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L91), [`_global_scalar_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L98), [`_scalar_store`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L104), [`_layout_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L111), [`_buffer_rsrc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L115), [`_lds_swizzle_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L121), [`lds_swizzle_mask_f8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L127), [`lds_dma_dst`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L132), [`global_typed_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L141), [`_global_i32_buffer_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L151), [`_global_i32_buffer_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L166), [`lds_typed_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L172), [`lds_vec_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L182), [`lds_dma_atom_128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L189), [`flat_buffer_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L194), [`_fabs_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L214), [`_e8m0_roundup`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L218), [`_e8m0_from_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L225), [`_inline_e8m0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L231), [`_pkmax_u16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L236), [`_clamp_gate_up_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L242), [`_swiglu_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L250), [`_situ_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L259), [`_activation_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L275), [`_umax_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L298), [`_dpp_umax_step`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L303), [`_inline_dpp_quad_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L308), [`_inline_dpp_pair_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L312), [`k_half_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L316), [`k_tiles_total_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L320), [`kunroll_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L324), [`kas_c_k1_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L328), [`kbs_c_k1_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L334), [`kbs_stride_n0_dw_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L338), [`kas_per_chunk_dw_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L342), [`num_n_blocks_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L346), [`kbs_c_n1_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L350), [`kbs_per_expert_dw_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L354), [`bq_bytes_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L358), [`bscale_bytes_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L362), [`kmchunks_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L366), [`lds_acc_bytes_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L370), [`fp8out_scale_blk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L379), [`fp8out_row_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py#L391).

### `aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py) · **Recent**

Source documentation: 8-wave MXFP8 matmul (per-1x32 E8M0 scales) for AMD CDNA4 (gfx950 only). Ported from FlyDSL kernels/gemm/mxfp8_gemm_8wave.py (incl. ROCm/FlyDSL#1275, row-major scales). The K pipeline is the a8w8 ptpc 8-wave one (gemm_a8w8_8wave), whose loaders it shares; the MFMA carries the E8M0 scales, so the epilogue only converts. MxMfma issues one scaled atom per 16x16 tile with scale_a= / scale_b=, the form the pinned FlyDSL takes.

Symbols: [`ScalePreshuffledS2R`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py#L36), [`ScaleRawLds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py#L94), [`MxMfma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py#L262), [`StoreBf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py#L320), [`compile_mxfp8_gemm_8w`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py#L361).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/mxfp4_gemm2_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py)

Symbols: [`_epilog_of`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L26), [`_get_compiled_mxfp4_gemm2_port`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L35), [`_dummy_out_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L64), [`_assert_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L68), [`flydsl_mxfp4_gemm2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L102).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py)
- [op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py)
- [op_tests/test_gemm_a16w8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w8_asm.py)
- [op_tests/test_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w4.py)
- [op_tests/test_gemm_a4w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w6.py)
- [op_tests/test_gemm_a6w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w4.py)
- [op_tests/test_gemm_a6w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w6.py)
- [op_tests/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8.py)
- [op_tests/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale.py)
- [op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py)
- [op_tests/test_gemm_a8w8_blockscale_fallback.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_fallback.py)
- [op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py)
- [op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py)
- [op_tests/test_mxfp8fp4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp8fp4gemm.py)
- [op_tests/test_quant_mxfp4_gemm_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4_gemm_fused.py)
- [op_tests/test_quant_mxfp6_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp6_gemm.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py)
- [op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py)

## Performance and validation

Compare identical quantized inputs and scale tensors. Count quantization, shuffle and reduction kernels when they execute on the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
