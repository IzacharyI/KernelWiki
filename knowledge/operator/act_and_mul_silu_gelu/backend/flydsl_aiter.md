---
title: "Activation and gated elementwise products: AITER flydsl"
kind: implementation_knowledge
operator: act_and_mul_silu_gelu
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Activation and gated elementwise products — AITER flydsl

Gated activation paths split or pair channels, apply SiLU/GELU or a clamped variant, then multiply. Some entries also scale or quantize the result. Small batches may require a different workgroup decomposition.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/act.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py) · **Recent**

Source documentation: Shared device-side gate/up activation helpers for FlyDSL MoE kernels. Elementwise f32-register helpers (exp2/rcp-based sigmoid, sign-restored tanh, and the gate*up batch forms) usable by any FlyDSL gemm1 fused gate+up epilog, plus :func:gate_up_act, which picks between them so kernels carry no act branch.

Symbols: [`sigmoid_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L21), [`sigmoid_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L27), [`clamp_gate_up`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L31), [`silu_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L36), [`swiglu_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L41), [`tanh_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L51), [`tanh_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L67), [`tanh_via_sigmoid_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L71), [`situ_mul`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L77), [`situ_mul_batch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L83), [`SituParams`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L105), [`situ_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L119), [`gate_up_act`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/act.py#L124).

### `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/silu_and_mul_fq.py) · **Recent**

Source documentation: Fused gate-activation-and-mul + quantization + sorted-scale write kernel (FlyDSL). Split-K MOE stage1 post-processing. Inputs: tmp_out (token_num*topk, inter_dim*2) bf16, optional topk_ids i32 / bias (expert, inter_dim*2) f32; sorted_token_ids (sorted_len,) i32 packed (token<<0 \| slot<<24) + num_valid_ids. Output: out (FP4x2, FP8, or BF16 per quant_mode) + out_scale_sorted tiled E8M0 scale (fp4/fp8 only).

Symbols: [`build_silu_and_mul_fq_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/silu_and_mul_fq.py#L34).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/swiglu_and_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/swiglu_and_mul.py) · **Recent**

Source documentation: Fused swiglu_and_mul kernel for interleaved (N0, 2, NLANE) layout (FlyDSL). Input (from cktile split_k with a16w4 interleave preshuffle): each row has inter_dim*2 bf16 columns laid out as [gate_b0(NLANE), up_b0(NLANE), gate_b1(NLANE), up_b1(NLANE), ...] i.e. shape (N0, 2, NLANE) with N0 = inter_dim // NLANE, NLANE = 16.

Symbols: [`build_swiglu_and_mul_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/swiglu_and_mul.py#L35).

Parameters referenced in this file: `split_k`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_silu_and_mul_fq.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_silu_and_mul_fq.py)
- [op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_clamp_act_mul.py)
- [op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_silu_and_mul_backward.py)
- [op_tests/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_activation.py)
- [op_tests/test_mxfp_exact_shape_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp_exact_shape_dispatch.py)
- [op_tests/triton_tests/fusions/test_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_activation.py)
- [op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py)
- [op_tests/triton_tests/fusions/test_fused_silu_mul.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_silu_mul.py)
- [op_tests/triton_tests/fusions/test_silu_and_mul_backward.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_silu_and_mul_backward.py)
- [op_tests/triton_tests/torch_compile/test_compile_activation.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_activation.py)

## Performance and validation

Match the activation approximation, split axis, optional scale and output dtype. Measure the actual short-batch regime before tuning vector widths.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
