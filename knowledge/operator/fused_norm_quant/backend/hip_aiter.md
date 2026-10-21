---
title: "Fused normalization and quantization: AITER hip"
kind: implementation_knowledge
operator: fused_norm_quant
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused normalization and quantization — AITER hip

Normalization, optional gates/residuals and scale generation share memory reads. Per-token and per-group scales have different output layouts and reduction domains.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/fused_norm_quant_gemm.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fused_norm_quant_gemm.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/gated_rmsnorm_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/gated_rmsnorm_quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/rmsnorm_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/rmsnorm_quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/fused_norm_quant_gemm.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_norm_quant_gemm.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/fused_qk_rmsnorm_group_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qk_rmsnorm_group_quant.cu) · **Recent**

Symbols: [`fused_qk_rmsnorm_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qk_rmsnorm_group_quant.cu#L31).

### `csrc/kernels/gated_rmsnorm_quant_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/gated_rmsnorm_quant_kernels.cu) · **Recent**

Symbols: [`gated_rmsnorm_fp8_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/gated_rmsnorm_quant_kernels.cu#L43), [`gated_rmsnorm_fp8_per_token_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/gated_rmsnorm_quant_kernels.cu#L344).

### `csrc/kernels/rmsnorm/rmsnorm_opus_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm/rmsnorm_opus_quant.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/rmsnorm_quant_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm_quant_kernels.cu) · **Recent**

Symbols: [`add_rmsnorm_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/rmsnorm_quant_kernels.cu#L23).

### `csrc/include/fused_qk_rmsnorm_group_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fused_qk_rmsnorm_group_quant.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/triton_test/test_fused_rs_rmsnorm_quant_ag.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/triton_test/test_fused_rs_rmsnorm_quant_ag.py)
- [op_tests/test_fused_kv_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_kv_norm_rope_group_quant.py)
- [op_tests/test_fused_qk_rmsnorm_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_rmsnorm_group_quant.py)
- [op_tests/test_fused_qk_rmsnorm_per_token_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_rmsnorm_per_token_quant.py)
- [op_tests/test_gated_rmsnorm_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gated_rmsnorm_fp8_quant.py)
- [op_tests/test_layernorm2dFusedAddQuant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_layernorm2dFusedAddQuant.py)
- [op_tests/test_rmsnorm2dFusedAddQuant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rmsnorm2dFusedAddQuant.py)

## Performance and validation

Verify normalized values, quantized output and scales together. Use the exact rounding, clipping and FP8/MXFP format from the reference.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
