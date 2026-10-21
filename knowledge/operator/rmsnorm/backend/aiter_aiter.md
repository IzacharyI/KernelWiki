---
title: "RMS normalization: AITER aiter"
kind: implementation_knowledge
operator: rmsnorm
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# RMS normalization — AITER aiter

RMSNorm reduces squares across a row, forms an inverse RMS and scales by a learned weight. Row width, number of rows and vectorized access determine the balance between launch cost, parallelism and memory traffic.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py) · **Recent**

Source documentation: out = rmsnorm(input) * (weight [+ 1 if gemma_norm]) (fp32 accumulate).

Symbols: [`_rms_norm_opus_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L18), [`_fused_add_rms_norm_opus_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L34), [`_add_rms_norm_opus_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L49), [`_check`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L66), [`rms_norm_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L80), [`fused_add_rms_norm_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L114), [`rmsnorm2d_fwd_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L143), [`rmsnorm2d_fwd_with_add_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L151), [`_rms_norm_quant_opus_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L201), [`_qmax_outcode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L221), [`_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L231), [`rmsnorm2d_fwd_with_dynamicquant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L269), [`rmsnorm2d_fwd_with_smoothquant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L285), [`rmsnorm2d_fwd_with_add_dynamicquant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L301), [`rmsnorm2d_fwd_with_add_smoothquant_opus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L326), [`rms_norm_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L356), [`fused_add_rms_norm_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L369), [`_rms_norm_fwd_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L379), [`_use_hip_common`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L388), [`_rms_norm_fwd_dispatch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L401), [`rms_norm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L416), [`rmsnorm2d_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L427), [`rmsnorm2d_fwd_with_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L439), [`rmsnorm2d_fwd_with_smoothquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L466), [`rmsnorm2d_fwd_with_add_smoothquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L480), [`rmsnorm2d_fwd_with_dynamicquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L506), [`rmsnorm2d_fwd_with_add_dynamicquant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L535), [`add_rmsnorm_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L586), [`add_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L602), [`rmsnorm_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L614), [`rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rmsnorm.py#L636).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_rmsnorm2d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rmsnorm2d.py)
- [op_tests/op_benchmarks/triton/bench_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_rmsnorm.py)
- [op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_rmsnorm_indexed_adaln.py)
- [op_tests/triton_tests/normalization/test_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/normalization/test_rmsnorm.py)
- [op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py)

## Performance and validation

Check accumulation dtype, epsilon, weight dtype and strided/tail rows. Benchmark decode-size row counts separately from prefill.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
