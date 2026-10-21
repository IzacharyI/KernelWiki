---
title: "Batched matrix multiplication: AITER ck"
kind: implementation_knowledge
operator: batched_gemm
backend: ck
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Batched matrix multiplication — AITER ck

Batch strides and batch size affect dispatch as well as M/N/K. A batch axis can expose parallelism that a single small GEMM lacks. Quantized variants add per-batch, per-token or block scale layouts.

CK implementations select generated/template instances. The selected instance, layout and specialization determine behavior; compare through the same public wrapper and include required packing or reduction.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/ck_batched_gemm_a8w8/include/batched_gemm_a8w8.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_a8w8/include/batched_gemm_a8w8.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py)
- [op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py)
- [op_tests/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8.py)
- [op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py)
- [op_tests/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_bf16.py)
- [op_tests/test_flydsl_batched_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_batched_gemm.py)
- [op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py)

## Performance and validation

Use the wrapper and matching reference for that variant. Report batch size, all matrix strides and any broadcasted operands.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](ck.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
