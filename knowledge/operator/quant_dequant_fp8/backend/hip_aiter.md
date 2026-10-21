---
title: "Quantization and dequantization: AITER hip"
kind: implementation_knowledge
operator: quant_dequant_fp8
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Quantization and dequantization — AITER hip

Quantization combines reduction for scales with clipping, rounding and conversion. Token, group and block variants have different scale tensors and alignment requirements.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/dsv4_dequant_gather_k.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/dsv4_dequant_gather_k.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/dsv4_rotate_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/dsv4_rotate_quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mx_quant_utils.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h) · **Recent**

Symbols: [`fp_f32_to_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L132), [`inv_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L202), [`fp_f32_to_e8m0_block_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L213), [`fp4_f32_to_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L225), [`mx_scale_shuffle_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L234), [`fp4_scale_shuffle_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mx_quant_utils.h#L241).

### `csrc/include/quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/dsv4_dequant_gather_k.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_dequant_gather_k.cu) · **Recent**

Symbols: [`fp8_e4m3_to_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_dequant_gather_k.cu#L43), [`f32_to_bf16_rne`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_dequant_gather_k.cu#L56), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_dequant_gather_k.cu#L62).

### `csrc/kernels/dsv4_rotate_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu) · **Recent**

Symbols: [`rotate_dim_rsqrt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L19), [`swap_thread_data`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L46), [`store_scale_e8m0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L90), [`kv_fp4_preshuffle_offset`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L131), [`kv_scale_preshuffle_offset`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L148), [`store_kv_scale_e8m0_preshuffle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L171), [`write_packed_fp4_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L185), [`store_kv_fp4_vector`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L202), [`store_kv_fp4_preshuffle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L232), [`hadamard_rotate_activation_fp4quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L265), [`rope_hadamard_rotate_activation_fp4quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L520), [`rope_hadamard_rotate_activation_fp8quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L902), [`norm_rope_hadamard_rotate_activation_fp4quant_kvcache_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/dsv4_rotate_quant.cu#L1140).

### `csrc/kernels/quant_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu) · **Recent**

Symbols: [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L61), [`initializeScale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L663), [`data_to_per_row_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L673), [`atomicMaxFloat`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L737), [`data_to_scale_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L749), [`scaled_quant_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L761), [`scaled_quant_vgpr_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L820), [`scaled_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L857), [`dynamic_per_token_scaled_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L866), [`smooth_data_to_per_row_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L912), [`smooth_per_token_scaled_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L950), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L1362), [`partial_transpose_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L1879), [`moe_smooth_per_token_scaled_quant_kernel_v1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L2010), [`moe_smooth_per_token_scaled_quant_kernel_v2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L2227), [`store_zero_mx_quant_moe_sort_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L2451), [`fused_mx_quant_moe_sort_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L2493), [`mxfp4_moe_sort_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_kernels.cu#L2832).

### `csrc/include/smoothquant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/smoothquant.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

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

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
