---
title: "Scaled and mixed-precision GEMM: AITER opus"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER opus

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/opus/gemm_op_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py) · **Recent**

Source documentation: Private OPUS A8W8 exact-kid launch APIs.

Symbols: [`_gen_opus_gemm_a8w8_launch_fake_tensors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L43), [`_opus_gemm_a8w8_launch_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L58), [`_gen_opus_gemm_a8w8_blockscale_launch_fake_tensors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L66), [`_opus_gemm_a8w8_blockscale_launch_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L83), [`_gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L93), [`_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L110), [`opus_gemm_a8w8_blockscale_bpreshuffle_tune`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L120), [`_gen_opus_gemm_a8w8_mxscale_bmm_launch_fake_tensors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L146), [`_opus_gemm_a8w8_mxscale_bmm_launch_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L165), [`_launch_a8w8_backend`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L177), [`_launch_a8w8_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L246), [`_launch_a8w8_blockscale_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L283), [`_launch_a8w8_blockscale_bpreshuffle_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L323), [`_validate_a8w8_mxscale_bmm_tensors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L369), [`_launch_a8w8_mxscale_bmm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L438), [`bmm_a8w8_mxscale_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L563), [`bmm_a8w8_mxscale_opus_bpreshuffle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L620), [`_launch_bmm_a8w8_mxscale_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L689), [`_mxscale_bmm_workspace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/gemm_op_a8w8.py#L716).

Parameters referenced in this file: `splitK`, `split_k`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py)
- [op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py)
- [op_tests/test_gemm_a16w8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w8_asm.py)
- [op_tests/test_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w4.py)
- [op_tests/test_gemm_a4w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w6.py)
- [op_tests/test_gemm_a6w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w4.py)
- [op_tests/test_gemm_a6w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w6.py)
- [op_tests/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8.py)
- [op_tests/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale.py)
- [op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py)
- [op_tests/test_gemm_a8w8_blockscale_fallback.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_fallback.py)
- [op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py)
- [op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py)
- [op_tests/test_mxfp8fp4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp8fp4gemm.py)
- [op_tests/test_quant_mxfp4_gemm_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4_gemm_fused.py)
- [op_tests/test_quant_mxfp6_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp6_gemm.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py)
- [op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py)

## Performance and validation

Compare identical quantized inputs and scale tensors. Count quantization, shuffle and reduction kernels when they execute on the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
