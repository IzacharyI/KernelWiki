---
title: "Activation and gated elementwise products: AITER triton"
kind: implementation_knowledge
operator: act_and_mul_silu_gelu
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Activation and gated elementwise products — AITER triton

Gated activation paths split or pair channels, apply SiLU/GELU or a clamped variant, then multiply. Some entries also scale or quantize the result. Small batches may require a different workgroup decomposition.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/activation.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py) · **Recent**

Source documentation: Apply the activation function and quantize the result to MX FP4 format. Args: x: The input tensor, typically fp16 or bf16. activation: activation function to apply before quantization. - It splits the features into two parts and applies the activation to the first part. - Then, it adds the results together before quantization. - Supports the following activations: - "silu" - "gelu" - "gelu_tanh" scaling_mode: The method to calculate MX block scaling.

Symbols: [`_get_silu_and_mul_backward_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py#L25), [`act_mul_and_mxfp4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py#L31), [`act_mul_and_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py#L145), [`fused_silu_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py#L219), [`silu_and_mul_backward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/activation.py#L328).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/fusions/fused_clamp_act_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py) · **Recent**

Source documentation: Fusion of chunk + activation + multiply + quantize, optional FP8 quant/shuffle. Splits inp into two halves, gate and up, then computes: out = act(clamp(gate)) * clamp(up) * weights (clamp and weights if applicable) Args: inp: input, shape [M, 2N], contiguous.

Symbols: [`_is_gluon_available`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py#L27), [`_pick_smallest_ceil`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py#L32), [`_load_default`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py#L41), [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py#L53), [`fused_clamp_act_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_clamp_act_mul.py#L86).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/common/activation.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py) · **Recent**

Symbols: [`_silu_exp2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L6), [`_silu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L11), [`_sigmoid_exp2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L16), [`_sigmoid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L21), [`_tanh`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L26), [`_gelu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L31), [`_gelu_tanh`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L38), [`_relu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L49), [`_relu6`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L54), [`_get_activation_from_str`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L58), [`_apply_activation_from_str`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/activation.py#L73).

### `aiter/ops/triton/_triton_kernels/fusions/activation.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/activation.py) · **Recent**

Source documentation: SiLU on the first half of the last dimension, multiply by the second half. Each row has 2 * n_cols input elements; writes n_cols outputs. 2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).

Symbols: [`fused_silu_mul_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/activation.py#L15), [`_silu_and_mul_backward_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/activation.py#L63), [`_act_mul_and_dynamic_mxfp4_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/activation.py#L133), [`_act_mul_and_dynamic_fp8_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/activation.py#L254).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py) · **Recent**

Source documentation: Fused SwiGLU clamp + SiLU * up + optional token weights + optional per-row FP8 group quant (128). Each program handles one row (token). inp is [M, 2 * N] with gate in the first N columns and up in the second N (same layout as torch.chunk(2, dim=-1)). Gate clamp matches DeepSeek-V4 reference: clamp(gate, max=limit) only; up uses clamp(up, min=-limit, max=limit). When HAS_QUANT is False the result is written directly to out in its native dtype and no scales are produced.

Symbols: [`_fused_clamp_silu_mul_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py#L37).

Parameters referenced in this file: `BLOCK_SIZE_N`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/triton/fusions/fused_clamp_act_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/fusions/fused_clamp_act_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/fusions/fused_clamp_act_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/fusions/fused_clamp_act_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=2048.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/silu_and_mul_backward/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/silu_and_mul_backward/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py)
- [op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py)
- [op_tests/triton_tests/fusions/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_activation.py)
- [op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py)
- [op_tests/triton_tests/fusions/test_fused_silu_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_silu_mul.py)
- [op_tests/triton_tests/fusions/test_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_silu_and_mul_backward.py)
- [op_tests/triton_tests/torch_compile/test_compile_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_activation.py)
- [op_tests/flydsl_tests/test_silu_and_mul_fq.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_silu_and_mul_fq.py)
- [op_tests/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_activation.py)
- [op_tests/test_mxfp_exact_shape_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp_exact_shape_dispatch.py)

## Performance and validation

Match the activation approximation, split axis, optional scale and output dtype. Measure the actual short-batch regime before tuning vector widths.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
