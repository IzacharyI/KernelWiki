---
title: "Elementwise: AITER triton"
kind: implementation_knowledge
operator: elementwise
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Elementwise — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_indexed_gated_residual.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_indexed_gated_residual.py) · **Recent**

Source documentation: Gated residual add with a table-indexed gate, in one pass. out[m] = residual[m] + gate[indices[m]] * x[m] This is the residual step that closes an attention or MLP branch of a diffusion transformer block, the counterpart of fused_rmsnorm_indexed_adaln that opens it: every token carries an index into a small table of gate vectors -- one row per (modality, timestep) combination.

Symbols: [`fused_indexed_gated_residual`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_indexed_gated_residual.py#L16).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/fusions/fused_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_mul_add.py) · **Recent**

Source documentation: Computes elementwise multiplicated and addtion: out = x * a + b Key parameters: - x: must be a torch.Tensor, but with arbitrary shape, - a: can be float, int, or torch.Tensor with shape (1, ) or the same shape as x - b: can be float, int, or torch.Tensor with shape (1, ) or the same shape as x all tensors must be contiguous if out is None, the kernel will perform inplace computation on x instead of creating a new tensor Returns: - out: same shape as x

Symbols: [`fused_mul_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_mul_add.py#L10).

Parameters referenced in this file: `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/fusions/fused_sigmoid_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_sigmoid_mul.py) · **Recent**

Source documentation: Replaces the eager two-kernel x * gate.sigmoid() (one sigmoid pass writing a temporary, one multiply pass reading it back) with a single pass

Symbols: [`_is_row_strided_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_sigmoid_mul.py#L23), [`fused_sigmoid_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_sigmoid_mul.py#L27).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_indexed_gated_residual.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_indexed_gated_residual.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_indexed_gated_residual.py#L20), [`_fused_indexed_gated_residual_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_indexed_gated_residual.py#L26).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_mul_add.py) · **Recent**

Symbols: [`_fused_mul_add_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_mul_add.py#L20).

Parameters referenced in this file: `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_sigmoid_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_sigmoid_mul.py) · **Recent**

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_sigmoid_mul.py#L17), [`_fused_sigmoid_mul_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_sigmoid_mul.py#L24).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/fusions/indexed_table.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/indexed_table.py) · **Recent**

Source documentation: Per-token gather from a small table of row vectors. Diffusion transformer blocks carry one index per token into a table with one row per (modality, timestep) combination. fused_rmsnorm_indexed_adaln reads its shift/scale rows this way when a branch opens, and fused_indexed_gated_residual its gate rows when the branch closes.

Symbols: [`_load_block_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/indexed_table.py#L17), [`_gather_indexed_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/indexed_table.py#L27).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1201/triton/fusions/fused_indexed_gated_residual/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/fusions/fused_indexed_gated_residual/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/fusions/fused_indexed_gated_residual/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/fusions/fused_indexed_gated_residual/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/fusions/fused_sigmoid_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/fusions/fused_sigmoid_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/fusions/fused_indexed_gated_residual/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/fusions/fused_indexed_gated_residual/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/fusions/fused_sigmoid_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/fusions/fused_sigmoid_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_indexed_gated_residual/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_indexed_gated_residual/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_sigmoid_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_sigmoid_mul/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_indexed_gated_residual.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_indexed_gated_residual.py)
- [op_tests/op_benchmarks/triton/bench_fused_sigmoid_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_sigmoid_mul.py)
- [op_tests/triton_tests/fusions/test_fused_indexed_gated_residual.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_indexed_gated_residual.py)
- [op_tests/triton_tests/fusions/test_fused_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_mul_add.py)
- [op_tests/triton_tests/fusions/test_fused_sigmoid_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_sigmoid_mul.py)
- [op_tests/triton_tests/torch_compile/test_compile_fused_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_fused_mul_add.py)
- [op_tests/test_aiter_sigmoid.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_aiter_sigmoid.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
