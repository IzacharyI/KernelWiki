---
title: "RMS normalization: AITER triton"
kind: implementation_knowledge
operator: rmsnorm
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# RMS normalization — AITER triton

RMSNorm reduces squares across a row, forms an inverse RMS and scales by a learned weight. Row width, number of rows and vectorized access determine the balance between launch cost, parallelism and memory traffic.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_rmsnorm_indexed_adaln.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_rmsnorm_indexed_adaln.py) · **Recent**

Source documentation: RMSNorm followed by a table-indexed affine modulation, in one pass. out[m] = rmsnorm(x[m], weight) * (1 + scale[indices[m]]) + shift[indices[m]] This is the adaptive-layernorm (AdaLN) step of a diffusion transformer. Every token carries an index into a small table of modulation vectors -- one row per (modality, timestep) combination -- so the affine parameters vary per token while the table stays tiny.

Symbols: [`fused_rmsnorm_indexed_adaln`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_rmsnorm_indexed_adaln.py#L16).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/normalization/rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py) · **Recent**

Source documentation: Applies Root Mean Square Layer Normalization over a mini-batch of inputs. Key parameters: - Input: The input tensor to be normalized with shape (M, N). - Weight: The learnable weights tensor with shape (N, ). - Epsilon: A value added to the denominator for numerical stability. Returns: - Output: The output tensor with shape (M, N).

Symbols: [`num_programs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L26), [`block_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L31), [`use_blocked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L47), [`dg_tmp_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L51), [`num_programs_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L55), [`_rmsnorm_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L71), [`_rmsnorm_forward_with_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L101), [`_rmsnorm_backward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L136), [`_should_use_large_m_small_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L238), [`rmsnorm_forward_inference`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L256), [`_RMSNorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L271), [`_RMSNorm2dFwdWithAdd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L306), [`rms_norm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L334), [`rmsnorm2d_fwd_with_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L352), [`rmsnorm2d_fwd_with_smoothquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L386), [`rmsnorm2d_fwd_with_dynamicquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L459), [`rmsnorm2d_fwd_with_add_smoothquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L534), [`rmsnorm2d_fwd_with_add_dynamicquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L604), [`_rmsnorm_forward_large_m_small_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py#L672).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/common/rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/rmsnorm.py) · **Recent**

Source documentation: Shared Triton RMSNorm operation for fused normalization kernels.

Symbols: [`_rmsnorm_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/rmsnorm.py#L11), [`_inverse_rms`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/rmsnorm.py#L17).

### `aiter/ops/triton/_triton_kernels/fusions/fused_rmsnorm_indexed_adaln.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_rmsnorm_indexed_adaln.py) · **Recent**

Symbols: [`_fused_rmsnorm_indexed_adaln_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_rmsnorm_indexed_adaln.py#L20).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py) · **Recent**

Symbols: [`_per_token_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L11), [`_rms_norm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L56), [`_quant_rms_norm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L189), [`_fused_add_rmsnorm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L430), [`_quant_fused_add_rmsnorm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L591), [`_rmsnorm_bwd_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L830), [`_rmsnorm_bwd_dg_reduce_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L981), [`_rmsnorm_kernel_large_m_small_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L1022), [`_rmsnorm_bwd_kernel_large_m_small_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py#L1077).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx942/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_rmsnorm.py)
- [op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py)
- [op_tests/triton_tests/normalization/test_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_rmsnorm.py)
- [op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py)
- [op_tests/test_rmsnorm2d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rmsnorm2d.py)

## Performance and validation

Check accumulation dtype, epsilon, weight dtype and strided/tail rows. Benchmark decode-size row counts separately from prefill.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
