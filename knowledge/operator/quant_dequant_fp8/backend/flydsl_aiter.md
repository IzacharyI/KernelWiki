---
title: "Quantization and dequantization: AITER flydsl"
kind: implementation_knowledge
operator: quant_dequant_fp8
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Quantization and dequantization — AITER flydsl

Quantization combines reduction for scales with clipping, rounding and conversion. Token, group and block variants have different scale tensors and alignment requirements.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/quant_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py) · **Recent**

Source documentation: Shared MX-format quantization IR helpers for FlyDSL kernels. These functions emit MLIR/LLVM IR for the per-block E8M0 scale calculation and the f32 -> fp4 (e2m1) conversion. They are *IR builders* -- you must call them inside an active InsertionPoint (i.e. while the FlyDSL DSL is mid-build of a kernel function), and they emit the same arith / LLVM ops the kernels would otherwise emit inline.

Symbols: [`emit_mx_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L77), [`emit_f32_to_e2m1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L186), [`emit_amax_e8m0_native_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L226), [`emit_cvt_scalef32_pk8_fp8_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L286), [`emit_cvt_scalef32_pk8_fp4_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L301), [`emit_cvt_scalef32_pk8_fp4_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quant_utils.py#L315).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py)
- [op_tests/op_benchmarks/triton/bench_quant_fp8_blockwise.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_fp8_blockwise.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp8.py)
- [op_tests/test_dsv4_rotate_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_dsv4_rotate_quant.py)
- [op_tests/test_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant.py)
- [op_tests/test_situv2_and_mul_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_situv2_and_mul_quant.py)
- [op_tests/test_smoothquant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_smoothquant.py)
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

## Performance and validation

Compare both payload and scales. Record FNUZ versus OCP formats, saturation behavior, group sizes and whether scale generation is included in timing.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
