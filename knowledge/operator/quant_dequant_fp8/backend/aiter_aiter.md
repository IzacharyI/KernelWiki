---
title: "Quantization and dequantization: AITER aiter"
kind: implementation_knowledge
operator: quant_dequant_fp8
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Quantization and dequantization — AITER aiter

Quantization combines reduction for scales with clipping, rounding and conversion. Token, group and block variants have different scale tensors and alignment requirements.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py) · **Recent**

Source documentation: Torch reference for MXFP4 (E2M1) per-1x32 block-scale quantization. Mirrors the HIP path (:func:per_1x32_f4_quant_hip / :func:quant_mxfp4_hip) and the Triton path (:func:per_1x32_f4_quant_triton); all share the E8M0 block layout and the :class:MxScaleRoundMode enum. PyTorch equivalent: torchao/prototype/mx_formats/{config,mx_tensor}.py. Args: x: Input of shape (..., N) or (M, N). scale: Pre-computed scale (optional, usually None). quant_dtype: Must be dtypes.fp4x2.

Symbols: [`__getattr__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L33), [`smoothquant_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L43), [`moe_smoothquant_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L49), [`get_dtype_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L56), [`pertoken_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L64), [`per_1x32_f4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L97), [`per_1x32_i4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L207), [`per_1x32_f4_quant_for_dot_scaled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L230), [`per_1x32_f8_scale_f8_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L272), [`per_tensor_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L318), [`per_block_quant_wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L331), [`get_torch_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L350), [`get_hip_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L368), [`get_triton_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L391), [`per_token_quant_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L407), [`per_group_quant_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L447), [`per_1x32_mx_quant_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L512), [`per_1x32_f4_quant_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L626), [`per_tensor_quant_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L654), [`per_token_quant_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L674), [`per_1x32_f4_quant_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L687), [`per_tensor_quant_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L694), [`get_torch_act`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L706), [`moe_smooth_per_token_scaled_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L717), [`static_per_tensor_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L775), [`dynamic_per_tensor_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L779), [`dynamic_per_token_scaled_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L783), [`dynamic_per_group_scaled_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L795), [`dynamic_per_group_scaled_quant_fp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L830), [`smooth_per_token_scaled_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L847), [`moe_smooth_per_token_scaled_quant_v1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L862), [`moe_smooth_per_token_scaled_quant_v2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L878), [`mxfp4_moe_sort_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L896), [`mxfp4_moe_sort_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L909), [`fused_dynamic_mx_quant_moe_sort_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L930), [`fused_dynamic_mx_quant_moe_sort_hip_bounded`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L949), [`quant_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L966), [`quant_mxfp4_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L979), [`fused_dynamic_mx_quant_moe_sort`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1088), [`fused_dynamic_mxfp4_quant_moe_sort`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1210), [`fused_dynamic_mxfp8_quant_moe_sort`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1243), [`partial_transpose`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1287), [`_rotate_activation_fp4quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1297), [`_rotate_activation_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1308), [`rotate_activation`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1315), [`_rope_rotate_activation_fp4quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1346), [`_rope_rotate_activation_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1364), [`_rope_rotate_activation_fp8quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1379), [`rope_rotate_activation`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1396), [`rmsnorm_rope_rotate_activation_fp4quant_kvcache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quant.py#L1459).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py)
- [op_tests/test_dsv4_rotate_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_dsv4_rotate_quant.py)
- [op_tests/test_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant.py)
- [op_tests/test_situv2_and_mul_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_situv2_and_mul_quant.py)
- [op_tests/test_smoothquant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_smoothquant.py)
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

## Performance and validation

Compare both payload and scales. Record FNUZ versus OCP formats, saturation behavior, group sizes and whether scale generation is included in timing.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
