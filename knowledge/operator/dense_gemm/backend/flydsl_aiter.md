---
title: "Dense matrix multiplication: AITER flydsl"
kind: implementation_knowledge
operator: dense_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER flydsl

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py) · **Recent**

Source documentation: gfx1250 (WMMA) backend for the FlyDSL a8w8 bpreshuffle GEMM.

Symbols: [`_lazy_import`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py#L28), [`_as_1d_fp32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py#L48), [`run_preshuffle_gemm_a8_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py#L60), [`parse_wmma_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py#L220), [`run_gemm_a8w8_bpreshuffle_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py#L226).

Parameters referenced in this file: `BLOCK_K`, `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_a16w16_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py) · **Recent**

Source documentation: gfx1250 FlyDSL backend for the A16W16 GEMM: torch-facing wrapper.

Symbols: [`_lazy_import`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py#L20), [`_p`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py#L41), [`_cached_launcher`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py#L79), [`_split_k_counters`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py#L87), [`gemm_a16w16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py#L92).

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a16w16_policy.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py) · **Recent**

Source documentation: Shape-aware policy selection for the FlyDSL A16W16 GEMM.

Symbols: [`GemmConfigPruner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py#L25), [`gfx1250_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py#L149), [`_gfx1250_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py#L157), [`get_flydsl_a16w16_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py#L199).

Parameters referenced in this file: `split_k`.

### `aiter/ops/flydsl/gemm_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py) · **Recent**

Source documentation: High-level FlyDSL GEMM APIs.

Symbols: [`_normalize_dtype_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L48), [`flydsl_hgemm_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L58), [`get_flydsl_hgemm_kernel_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L89), [`flydsl_hgemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L119), [`_get_compile_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L209), [`_get_preshuffle_split_buffers`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L234), [`_check_preshuffle_flat_buffer_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L250), [`_check_preshuffle_split_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L272), [`flydsl_preshuffle_gemm_a8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py#L289).

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gather_gemm_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py) · **Recent**

Source documentation: 8-wave fused gather + kv_b_proj expansion for DeepSeek MLA on CDNA4 (gfx950).

Symbols: [`_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L42), [`lds_block_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L46), [`_gather_a_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L56), [`_WideG2SLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L93), [`BlockScale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L132), [`_pack_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L188), [`StoreKV`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L200), [`_RopeCopy`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L415), [`compile_gather_kv_b_proj_8w`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py#L565).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py) · **Recent**

Source documentation: Compute C[M, N] = A[M, K] @ B[K, N]. Each layout character controls only the corresponding tensor stride: N is row-major and T is column-major. Logical tensor shapes never change. Inputs that violate the selected layout or DMA alignment are rejected. Set user_kwargs["split_k"] above 1 to atomically reduce K partitions. Set user_kwargs["k_waves"] above 1 for full-tile workgroup-local slice-K. out_dtype may be the input dtype or torch.float32.

Symbols: [`GemmA16W16Gfx950Param`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L37), [`GemmABLoadContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L68), [`AsyncLoadTile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L81), [`make_gemm_a16w16_gfx950_param`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L92), [`make_gemm_a16w16_gfx950_kernel_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L295), [`make_gemm_ab_lds_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L312), [`make_gemm_ab_load_context`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L326), [`async_load_to_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L368), [`write_cshuffle_vec_to_global`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L427), [`gemm_a16w16_gfx950_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L484), [`gemm_a16w16_hti_gfx950_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L852), [`gemm_a16w16_gfx950`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1280), [`make_gemm_a16w16_param_and_validate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1359), [`assert_no_k_tail`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1392), [`get_split_k_buffers`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1438), [`_dynamic_tensor_arg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1451), [`gemm_a16w16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py#L1457).

Parameters referenced in this file: `split_k`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py) · **Recent**

Symbols: [`wait_vmcnt_and_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L21), [`get_llvm_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L26), [`_global_sync_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L33), [`store_global_f32_vec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L41), [`BlockSwizzle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L60), [`SplitKProtocol`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L95), [`make_lds_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L249), [`make_transposed_lds_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L257), [`get_wave_lds_offset`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L273), [`swizzled_col_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L280), [`transposed_contiguous_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py#L285).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_THREADS`, `split_k`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py) · **Recent**

Symbols: [`_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py#L23), [`apply_activation_scalar`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py#L27), [`compile_gemm_a16w16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py#L44).

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py) · **Recent**

Source documentation: Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8).

Symbols: [`make_lds_copy_ops`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L15), [`make_sgpr_opaque`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L45), [`_lds_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L68), [`lds_load_b128_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L75), [`lds_load_b32_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L82), [`lds_store_b128_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L88), [`workgroup_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L94), [`pipeline_fence`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L107), [`pipeline_fence_signal`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L119), [`pipeline_fence_wait`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L127), [`fmin_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L134), [`fclamp_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L139), [`_fma_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L146), [`fused_silu_swiglu_elem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L151), [`_tanh_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L169), [`situv2_consts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L190), [`fused_situv2_elem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L211), [`batched_situv2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L225), [`batched_silu_swiglu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py#L278).

### `aiter/ops/flydsl/kernels/preshuffle_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/preshuffle_gemm.py) · **Recent**

Source documentation: Preshuffle GEMM (layout API): f16/bf16/fp8/int8, ping-pong scf.for loop with scheduler hints.

Symbols: [`_get_preload`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/preshuffle_gemm.py#L117), [`compile_preshuffle_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/preshuffle_gemm.py#L129).

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py)
- [op_tests/test_deepgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_deepgemm.py)
- [op_tests/test_f4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_f4gemm.py)
- [op_tests/test_flydsl_hgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hgemm.py)
- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w16.py)
- [op_tests/test_gemm_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_codegen.py)
- [op_tests/test_gemm_group32_interface.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_group32_interface.py)
- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_opus_a16w16_gemm.py)
- [op_tests/triton_tests/gemm/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/__init__.py)
- [op_tests/triton_tests/gemm/basic/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/__init__.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py)
- [op_tests/triton_tests/gemm/batched/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/__init__.py)
- [op_tests/triton_tests/gemm/fused/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/__init__.py)
- [op_tests/triton_tests/gemm/grouped/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/__init__.py)
- [op_tests/triton_tests/gemm/grouped/test_gmm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/test_gmm.py)
- [op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py)

## Performance and validation

Measure packing and split-K reduction separately, then include both in the deployed operator cost. Keep accumulation and epilogue semantics unchanged.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
