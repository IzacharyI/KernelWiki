---
title: "Activation and gated elementwise products: AITER gluon"
kind: implementation_knowledge
operator: act_and_mul_silu_gelu
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Activation and gated elementwise products — AITER gluon

Gated activation paths split or pair channels, apply SiLU/GELU or a clamped variant, then multiply. Some entries also scale or quantize the result. Small batches may require a different workgroup decomposition.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_clamp_act_mul.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_clamp_act_mul.py) · **Recent**

Source documentation: Gluon (gfx1250) port of triton _fused_clamp_silu_mul_kernel for GFX1250

Symbols: [`_fused_clamp_silu_mul_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_clamp_act_mul.py#L36).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=512.json)
- [aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_clamp_act_mul/FUSED_CLAMP_ACT_MUL-N=8192.json)

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

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
