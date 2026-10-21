---
title: "Quantization and dequantization: AITER triton"
kind: implementation_knowledge
operator: quant_dequant_fp8
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Quantization and dequantization — AITER triton

Quantization combines reduction for scales with clipping, rounding and conversion. Token, group and block variants have different scale tensors and alignment requirements.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/quant/fused_fp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py) · **Recent**

Source documentation: This op contains several steps: 1. if res1 is not None, inp1 = inp1 + res1, and store inp1 to out_res1 2. perform RMS norm along the last dimenion for inp1 3. if inp2 is not None, perform RMS norm along the last dimenion for inp2 4. perform fp8 quantization for inp1 only Key parameters: - x: Matrix X with shape (M, N1, N2). Returns: - out1_fp8: The output matrix with shape (M, N1). - out1_s: The output matrix with shape (1,). - out1: The output matrix with shape (M, N1).

Symbols: [`fused_rms_fp8_per_tensor_static_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L27), [`fused_rms_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L164), [`get_fp8_min_max_bounds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L367), [`_num_compute_units`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L377), [`calc_rows_per_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L382), [`fused_rms_gated_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L396), [`fused_flatten_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L533), [`fused_reduce_act_mul_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L606), [`fused_reduce_rms_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L734), [`fused_silu_mul_fp8_per_tensor_static_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_fp8_quant.py#L992).

Parameters referenced in this file: `BLOCK_G`, `BLOCK_SIZE_M2`, `BLOCK_SIZE_N`, `BLOCK_SIZE_N1`, `BLOCK_SIZE_N2`, `BLOCK_SIZE_N3`, `BLOCK_SIZE_PAD`, `num_warps`.

### `aiter/ops/triton/quant/fused_mxfp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_mxfp8_quant.py) · **Recent**

Source documentation: Fused RMSNorm + MXFP8 (1x32 e8m0) quant in a single Triton launch. Args: x: (M, K) bf16 or fp16. weight: (K,) bf16 or fp16 RMSNorm weight. eps: RMSNorm epsilon. y: optional preallocated FP8 e4m3fn output (M, K). scale: optional preallocated uint8 e8m0 output (M, K // 32). Returns: y (M, K) fp8 e4m3fn, scale (M, K // 32) uint8.

Symbols: [`fused_rms_mxfp8_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_mxfp8_quant.py#L22), [`fused_dual_rmsnorm_mxfp8_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_mxfp8_quant.py#L78), [`fused_flatten_mxfp8_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/fused_mxfp8_quant.py#L168).

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_KK`, `BLOCK_SIZE_KQ`, `BLOCK_SIZE_N2`.

### `aiter/ops/triton/quant/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py) · **Recent**

Source documentation: Quantizes tensor using the provided scale to int8 or fp8 Parameters: - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller - x_in: Input tensor of shape (M, N). - scale_in: Input Scale tensor of shape (1,) and dtype fp32 Returns: - qx: Quantized output values.

Symbols: [`_has_scaled_downcast`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L51), [`_use_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L63), [`static_per_tensor_quant_fp8_i8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L72), [`dynamic_per_tensor_quant_fp8_i8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L126), [`dynamic_per_token_quant_fp8_i8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L163), [`dynamic_mxfp4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L203), [`dynamic_mxfp4_quant_blockscale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L457), [`dynamic_mxfp8_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L518), [`dynamic_mxfp8_quant_n32k4_mbn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L674), [`fp8_legacy_to_mxfp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L733), [`dynamic_nvfp4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant.py#L794).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/quant/quant_fp8_blockwise.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py) · **Recent**

Source documentation: Python wrappers for FP8 block-wise quantization kernels.

Symbols: [`_launch_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L32), [`_check_block_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L40), [`quant_fp8_blockwise`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L55), [`quant_fp8_blockwise_segment_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L114), [`quant_fp8_blockwise_for_weight`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L175), [`quant_fp8_blockwise_for_act_grad`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L224), [`requant_fp8_row_to_col`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_fp8_blockwise.py#L278).

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/quant/quant_mxfp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_mxfp8.py) · **Recent**

Source documentation: Quantize a tensor to MXFP8 format. Args: x: Input tensor of shape (M, N), dtype float32 or bfloat16. fp8_dtype: Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2). quant_block_size: Block size for quantization scaling (default 32). is_2d_block: Whether to use 2D block scaling. use_sr: Whether to use stochastic rounding. use_asm: Whether to use the inline-assembly path.

Symbols: [`convert_to_mxfp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_mxfp8.py#L22), [`convert_from_mxfp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/quant_mxfp8.py#L129).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py) · **Recent**

Symbols: [`_rmsmorm_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L16), [`_fp8_quant_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L26), [`_fused_rms_fp8_per_tensor_static_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L57), [`_fused_rms_fp8_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L189), [`_fused_flatten_fp8_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L436), [`_fused_reduce_act_mul_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L505), [`_fused_reduce_rms_fp8_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L674), [`_fused_silu_mul_fp8_per_tensor_static_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L931).

Parameters referenced in this file: `BLOCK_G`, `BLOCK_SIZE_M`, `BLOCK_SIZE_M2`, `BLOCK_SIZE_N`, `BLOCK_SIZE_N1`, `BLOCK_SIZE_N2`, `BLOCK_SIZE_N3`.

### `aiter/ops/triton/_triton_kernels/quant/fused_mxfp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_mxfp8_quant.py) · **Recent**

Symbols: [`_fused_rms_mxfp8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_mxfp8_quant.py#L33), [`_fused_dual_rmsnorm_mxfp8_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_mxfp8_quant.py#L130), [`_fused_flatten_mxfp8_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/fused_mxfp8_quant.py#L242).

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_KK`, `BLOCK_SIZE_KQ`, `BLOCK_SIZE_N2`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/quant/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py) · **Recent**

Symbols: [`_static_per_tensor_quant_fp8_i8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L20), [`_dynamic_per_tensor_quant_fp8_i8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L71), [`_dynamic_per_token_quant_fp8_i8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L100), [`_mxfp4_scale_from_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L131), [`_mxfp4_ceil_scale_from_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L146), [`_mxfp4_quant_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L158), [`_mxfp4_pack_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L198), [`_mxfp4_pack_cvt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L216), [`_mxfp4_pack_bits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L235), [`_nvfp4_quant_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L309), [`_mxfp4_e8m0_to_fp32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L407), [`_mxfp4_sr_random_words`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L415), [`_mxfp4_sr_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L444), [`_dynamic_mxfp4_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L517), [`_dynamic_mxfp4_quant_blockscale_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L650), [`_mxfp8_quant_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L715), [`_dynamic_mxfp8_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L746), [`_dynamic_mxfp8_quant_n32k4_mbn_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L814), [`_fp8_legacy_to_mxfp8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L906), [`_dynamic_nvfp4_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant.py#L986).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py) · **Recent**

Symbols: [`_compute_scale_and_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py#L29), [`quant_fp8_blockwise_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py#L48), [`quant_fp8_blockwise_segment_m_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py#L112), [`quant_fp8_blockwise_for_weight_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py#L166), [`requant_fp8_row_to_col_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py#L220).

Parameters referenced in this file: `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py) · **Recent**

Symbols: [`_calculate_scales`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L34), [`_generate_randval`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L111), [`_pack_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L120), [`_unpack_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L256), [`_convert_to_mxfp8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L335), [`_convert_from_mxfp8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/quant_mxfp8.py#L407).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_quant_fp8_blockwise.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_fp8_blockwise.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp8.py)
- [op_tests/triton_tests/quant/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/__init__.py)
- [op_tests/triton_tests/quant/test_convert_mxfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_convert_mxfp8.py)
- [op_tests/triton_tests/quant/test_fused_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_fused_fp8_quant.py)
- [op_tests/triton_tests/quant/test_fused_rms_gated_fp8_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_fused_rms_gated_fp8_group_quant.py)
- [op_tests/triton_tests/quant/test_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant.py)
- [op_tests/triton_tests/quant/test_quant_configs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_configs.py)
- [op_tests/triton_tests/quant/test_quant_fp8_blockwise.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_fp8_blockwise.py)
- [op_tests/triton_tests/quant/test_quant_mxfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_mxfp8.py)
- [op_tests/triton_tests/torch_compile/test_compile_quant_per_tensor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_quant_per_tensor.py)
- [op_tests/triton_tests/torch_compile/test_compile_quant_per_token.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_quant_per_token.py)
- [op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py)
- [op_tests/test_dsv4_rotate_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_dsv4_rotate_quant.py)
- [op_tests/test_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant.py)
- [op_tests/test_situv2_and_mul_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_situv2_and_mul_quant.py)
- [op_tests/test_smoothquant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_smoothquant.py)

## Performance and validation

Compare both payload and scales. Record FNUZ versus OCP formats, saturation behavior, group sizes and whether scale generation is included in timing.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
