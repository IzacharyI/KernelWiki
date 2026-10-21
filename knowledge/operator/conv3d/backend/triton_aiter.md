---
title: "Three-dimensional convolution: AITER triton"
kind: implementation_knowledge
operator: conv3d
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Three-dimensional convolution — AITER triton

AITER contains implicit-GEMM, im2col, transpose and specialized 3D convolution paths. The cost of input preparation can dominate a fast inner GEMM.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/conv/conv3d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py) · **Recent**

Source documentation: Forward 3-D conv on AMD ROCm via Triton. Drop-in for the forward of torch.nn.functional.conv3d (no backward). Parameters ---------- x : Tensor Input with logical shape [N, C, D, H, W], fp16 or bf16. For layout="ndhwc" it may carry channels-last-3d strides (or be converted to them internally). w_oidhw : Tensor Weight in PyTorch-canonical [K_out, C, T, R, S] layout (T = depth tap). bias : Tensor, optional 1-D bias of length K_out; kernel loads promote it to fp32.

Symbols: [`_require_cuda_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L40), [`Route3D`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L49), [`_select_3x3x3_ncdhw_method`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L59), [`_resolve_route`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L122), [`conv3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L162), [`conv3d_general`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L228), [`conv3d_1x1x1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L284), [`conv3d_ndhwc_3x3x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L337), [`conv3d_ncdhw_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L384), [`conv3d_winograd_hw_f4x3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L450), [`conv3d_winograd_hw_f4x3_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L498), [`_route_and_run`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L564), [`conv3d_ncdhw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L629), [`conv3d_ndhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/conv3d.py#L656).

Parameters referenced in this file: `BLOCK_K`.

### `aiter/ops/triton/_triton_kernels/conv/conv3d_1x1x1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_1x1x1.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_1x1x1.py#L14), [`_conv3d_1x1x1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_1x1x1.py#L34).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py) · **Recent**

Symbols: [`_get_config_ndhwc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py#L14), [`_get_config_cblocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py#L23), [`_conv3d_3x3x3_ndhwc_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py#L45), [`_conv3d_3x3x3_cblocked_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_3x3x3.py#L196).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv3d_general.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_general.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_general.py#L14), [`_conv3d_general_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_general.py#L34).

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py) · **Recent**

Source documentation: 2.5D Winograd for 3x3x3 conv3d: F(4x4, 3x3) on the (H, W) plane with the depth axis handled as a direct 3-tap reduction. Pipeline (NCDHW, stride=1, pad, dilation=1): 1. input transform : V[36, N*D*T_hw, C_pad] = Bᵀ d B on every input (H,W) plane of every input depth slice 2. batched GEMM : M[36, N*OD*T_hw, K] = Σ_t Σ_c U[t] · V(depth-shifted), reducing over the 3 depth taps (id = od - pad_d + t, masked to [0, D)) and channels 3.

Symbols: [`_config_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L31), [`_get_config_input`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L35), [`_get_config_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L40), [`_get_config_output`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L45), [`_winograd_hw_f4x3_input_transform_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L64), [`_winograd_hw_f4x3_batched_gemm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L145), [`_winograd_hw_f4x3_output_transform_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/conv3d_winograd_hw_f4x3.py#L234).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_1x1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_1x1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_1x1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_1x1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_1x1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_1x1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_1x1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_1x1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_1x1x1/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_1x1x1/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_3x3x3_cblocked/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_3x3x3_ndhwc/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_general/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_general/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_prepack/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_prepack/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_wino_hw_f4x3_input/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/conv/conv3d_wino_hw_f4x3_output/DEFAULT.json)

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
