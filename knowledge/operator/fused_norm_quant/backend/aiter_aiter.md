---
title: "Fused normalization and quantization: AITER aiter"
kind: implementation_knowledge
operator: fused_norm_quant
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused normalization and quantization — AITER aiter

Normalization, optional gates/residuals and scale generation share memory reads. Per-token and per-group scales have different output layouts and reduction domains.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/fused_norm_quant_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_norm_quant_gemm.py) · **Recent**

Source documentation: Fused RMSNorm + FP8 Quant + GEMM host-level fusion. Calls rmsnorm_quant (CK kernel) then hipb_mm (hipBLASLt GEMM) back-to-back with minimal Python overhead between them. Falls back to separate Python calls if the fused C++ JIT module is not available.

Symbols: [`_get_fused_cpp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_norm_quant_gemm.py#L19), [`fused_rmsnorm_quant_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_norm_quant_gemm.py#L39).

### `aiter/ops/fused_qk_rmsnorm_group_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py) · **Recent**

Symbols: [`_fused_qk_rmsnorm_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py#L19), [`_fused_qk_rmsnorm_per_token_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py#L43), [`fused_qk_rmsnorm_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py#L60), [`fused_qk_rmsnorm_per_token_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py#L137), [`fused_qk_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_rmsnorm_group_quant.py#L179).

### `aiter/ops/gated_rmsnorm_fp8_group_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gated_rmsnorm_fp8_group_quant.py) · **Recent**

Source documentation: Fused Gated RMSNorm + FP8 Group Quantization Operations: 1. Per-head Gated RMSNorm: norm(x) * silu(z) where: - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm) - silu(z) = z / (1 + exp(-z)) 2. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim] 3. FP8 group quantization with group_size=128 Constraint: ONLY supports head_dim=128 and group_size=128

Symbols: [`gated_rmsnorm_fp8_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gated_rmsnorm_fp8_group_quant.py#L24).

### `aiter/ops/gated_rmsnorm_fp8_per_token_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gated_rmsnorm_fp8_per_token_quant.py) · **Recent**

Source documentation: Fused Gated RMSNorm + FP8 Per-Token Quantization Operations: 1. Per-head Gated RMSNorm: norm(x) * gate(z) where: - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm over head_dim) - gate(z) = sigmoid(z) if use_sigmoid, else silu(z) - sigmoid(z) = 1 / (1 + exp(-z)) - silu(z) = z / (1 + exp(-z)) 2. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim] 3. FP8 per-token quantization: ONE scale per token across the full flattened row.

Symbols: [`gated_rmsnorm_fp8_per_token_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gated_rmsnorm_fp8_per_token_quant.py#L30).

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

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
