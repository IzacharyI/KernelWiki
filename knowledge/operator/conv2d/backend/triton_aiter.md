---
title: "Conv2d: AITER triton"
kind: implementation_knowledge
operator: conv2d
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Conv2d — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/conv/_launch.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py) · **Recent**

Symbols: [`_kernel_activation`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L96), [`_make_mn_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L101), [`_make_wino_input_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L113), [`_make_wino_gemm_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L123), [`_make_wino_output_grid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L136), [`_launch_nchw_to_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L146), [`_launch_1x1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L170), [`_launch_3x3_nhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L242), [`_launch_3x3_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L313), [`_launch_3x3_nchw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L386), [`_launch_general`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L454), [`_launch_winograd_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L535), [`_launch_ncdhw_to_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L654), [`_launch_general_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L678), [`_launch_1x1x1_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L766), [`_launch_3x3x3_ndhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L841), [`_launch_3x3x3_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L920), [`_launch_winograd_hw_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_launch.py#L1001).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/conv/_prepack.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py) · **Recent**

Source documentation: Weight/input repacking for convolution kernels. Several conv kernels don't consume the raw OIHW weight (or NCHW input) layout directly — they need it reshaped into a kernel-local format for coalesced loads: K-major padded tiles for the 1x1/general GEMM, [K_out, 9, C_pad] for the 3x3 kernels, channel-blocked NCHWc for the cblocked path, and the G·g·Gᵀ filter transform for Winograd F(4x4,3x3).

Symbols: [`_read_pack_cache_maxsize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L32), [`_LRUPackCache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L48), [`_transform_winograd_filter_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L95), [`clear_conv2d_weight_pack_caches`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L105), [`clear_conv3d_weight_pack_caches`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L112), [`_pack_cache_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L119), [`_round_up_to_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L133), [`_pad_last_dim`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L137), [`_prepack_to_kmajor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L145), [`_prepack_fixed_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L152), [`_get_or_make_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L160), [`prepack_oihw_to_kmajor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L173), [`get_or_make_weight_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L179), [`prepack_oihw_to_3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L183), [`get_or_make_weight_pack_3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L190), [`prepack_nchw_to_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L194), [`prepack_winograd_filter_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L211), [`get_or_make_winograd_filter_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L223), [`prepack_oidhw_to_kmajor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L237), [`get_or_make_weight_pack_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L244), [`prepack_oidhw_to_3x3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L250), [`get_or_make_weight_pack_3x3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L258), [`prepack_ncdhw_to_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L264), [`prepack_winograd_hw_filter_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L278), [`get_or_make_winograd_hw_filter_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_prepack.py#L294).

Parameters referenced in this file: `BLOCK_K`.

### `aiter/ops/triton/conv/_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py) · **Recent**

Symbols: [`_is_amd_wave32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L15), [`_out_hw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L21), [`_conv_dims`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L30), [`_alloc_output`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L40), [`_prep_bias`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L52), [`_ensure_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L62), [`_is_1x1_conv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L75), [`_is_3x3_conv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L80), [`_is_winograd_2d_eligible`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L85), [`_require_winograd_2d_eligible`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L93), [`_normalize_tuple`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L104), [`_normalize_conv2d_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L124), [`_normalize_conv3d_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L137), [`_out_dhw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L145), [`_conv3d_dims`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L156), [`_alloc_output_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L175), [`_is_1x1x1_conv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L187), [`_is_3x3x3_conv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L191), [`_is_winograd_hw_3d_eligible`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L195), [`_require_winograd_hw_3d_eligible`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L203), [`_winograd_transform_storage_dtype`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/_utils.py#L213).

Parameters referenced in this file: `BLOCK_K`.

### `aiter/ops/triton/conv/conv2d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py) · **Recent**

Source documentation: Forward 2-D conv on AMD ROCm via Triton. Drop-in for the forward of torch.nn.functional.conv2d (no backward). A shape-driven router picks among six kernel families (1x1, direct NCHW 3x3, cblocked 3x3, NHWC 3x3, Winograd F(4x4,3x3), general) per call. Inputs must be fp16 or bf16. layout="nhwc" runs an NHWC-native kernel with no internal layout conversion. Output dtype always matches the input dtype, matching torch.nn.Conv2d semantics.

Symbols: [`Route`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L48), [`_nchw_direct_is_profitable`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L67), [`_select_3x3_method`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L105), [`_resolve_route`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L131), [`conv2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L169), [`conv2d_winograd_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L225), [`conv2d_winograd_f4x3_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L264), [`conv2d_1x1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L318), [`conv2d_general`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L357), [`conv2d_nhwc_3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L400), [`_route_and_run`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L440), [`conv2d_nchw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L531), [`conv2d_nhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L558), [`conv2d_nchw_3x3_direct`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L590), [`conv2d_nchw_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv2d.py#L632).

Parameters referenced in this file: `BLOCK_K`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/conv/conv_1x1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_1x1.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_1x1.py#L14), [`_conv2d_1x1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_1x1.py#L33).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv_3x3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py) · **Recent**

Symbols: [`_get_config_nhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L14), [`_get_config_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L18), [`_get_config_nchw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L24), [`_conv2d_3x3_nhwc_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L69), [`_conv2d_3x3_cblocked_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L191), [`_conv2d_3x3_nchw_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3.py#L330).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py) · **Recent**

Symbols: [`_get_config_input`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L15), [`_get_config_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L19), [`_get_config_output`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L23), [`_winograd_f4x3_input_transform_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L52), [`_winograd_f4x3_cblocked_input_transform_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L115), [`_winograd_f4x3_batched_gemm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L175), [`_winograd_f4x3_output_transform_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py#L244).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv_general.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_general.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_general.py#L14), [`_conv2d_general_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv_general.py#L33).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py#L11), [`_get_config_3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py#L15), [`_ncx_to_cblocked_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/ncx_to_cblocked.py#L29).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_M`.

### `aiter/ops/triton/_triton_kernels/conv/winograd_f4x3_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/winograd_f4x3_common.py) · **Recent**

Source documentation: Shared F(4x4, 3x3) Winograd input and output transforms.

Symbols: [`_winograd_f4x3_input_transform`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/winograd_f4x3_common.py#L15), [`_winograd_f4x3_output_transform`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/winograd_f4x3_common.py#L328).

Parameters referenced in this file: `BLOCK_C`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_nchw/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_nchw/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1101/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_nchw/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_nchw/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_nchw/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_nchw/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_nchw/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_nchw/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv_wino_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_3x3_nhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_3x3_nhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_wino_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_wino_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv_wino_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv_wino_f4x3_output/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_conv2d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_conv2d.py)
- [op_tests/triton_tests/conv/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/__init__.py)
- [op_tests/triton_tests/conv/_helpers.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/_helpers.py)
- [op_tests/triton_tests/conv/test_conv2d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_conv2d.py)
- [op_tests/test_flydsl_conv_implicit.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_conv_implicit.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
