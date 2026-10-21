---
title: "Activation and gated elementwise products: AITER aiter"
kind: implementation_knowledge
operator: act_and_mul_silu_gelu
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Activation and gated elementwise products — AITER aiter

Gated activation paths split or pair channels, apply SiLU/GELU or a clamped variant, then multiply. Some entries also scale or quantize the result. Small batches may require a different workgroup decomposition.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/activation.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py) · **Recent**

Source documentation: Apply SiTUv2 and per-token FP8 quantization. All tensors must be contiguous ROCm tensors. input is BF16 with shape [..., 2 * d] where d is divisible by 8, out is FP8 with input.numel() / 2 elements, and scale is FP32 with one element per token. Only group_size == d and shuffle_scale=False are supported.

Symbols: [`silu_and_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L12), [`swiglu_and_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L16), [`silu_and_mul_bias`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L20), [`swiglu_and_mul_bias`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L26), [`gelu_and_mul_bias`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L32), [`scaled_silu_and_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L38), [`silu_and_mul_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L42), [`situv2_and_mul_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L53), [`gelu_and_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L72), [`gelu_tanh_and_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L76), [`gelu_fast`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L80), [`relu2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/activation.py#L84).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_activation.py)
- [op_tests/test_mxfp_exact_shape_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp_exact_shape_dispatch.py)
- [op_tests/flydsl_tests/test_silu_and_mul_fq.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_silu_and_mul_fq.py)
- [op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py)
- [op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py)
- [op_tests/triton_tests/fusions/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_activation.py)
- [op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py)
- [op_tests/triton_tests/fusions/test_fused_silu_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_silu_mul.py)
- [op_tests/triton_tests/fusions/test_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_silu_and_mul_backward.py)
- [op_tests/triton_tests/torch_compile/test_compile_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_activation.py)

## Performance and validation

Match the activation approximation, split axis, optional scale and output dtype. Measure the actual short-batch regime before tuning vector widths.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
