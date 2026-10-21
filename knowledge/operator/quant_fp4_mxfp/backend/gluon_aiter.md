---
title: "Packed low-bit and microscaling formats: AITER gluon"
kind: implementation_knowledge
operator: quant_fp4_mxfp
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Packed low-bit and microscaling formats — AITER gluon

Low-bit values are packed and interpreted together with scale metadata. Logical tensor sizes differ from storage sizes, and consumers may require shuffled scale/weight layouts.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py) · **Recent**

Symbols: [`_rmsmorm_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py#L14), [`_mxfp4_quant_op`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py#L27), [`_gluon_fused_rms_mxfp4_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py#L161), [`_gluon_fused_reduce_rms_mxfp4_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py#L465), [`_gluon_fused_dynamic_mxfp4_quant_moe_sort_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/quant/fused_mxfp4_quant.py#L925).

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_N2`, `BLOCK_SIZE_N3`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/quant/mxfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/quant/mxfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/quant/mxfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/quant/mxfp4/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py)
- [op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp4_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp4_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp4_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp4_fp8.py)
- [op_tests/test_iq2r_encoder.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_encoder.py)
- [op_tests/test_iq2r_format.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_format.py)
- [op_tests/test_iq2r_glm53.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm53.py)
- [op_tests/test_iq2r_glm5_calibrate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm5_calibrate.py)
- [op_tests/test_iq2r_glm5_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm5_compile.py)
- [op_tests/test_iq2r_hip.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_hip.py)
- [op_tests/test_iq2r_reference.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_reference.py)
- [op_tests/test_quant_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4.py)
- [op_tests/triton_tests/quant/test_fused_mxfp4_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_fused_mxfp4_quant.py)
- [op_tests/triton_tests/quant/test_mxfp4_jit_import.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_mxfp4_jit_import.py)
- [op_tests/triton_tests/quant/test_quant_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_mxfp4.py)
- [op_tests/triton_tests/quant/test_quant_mxfp4_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_mxfp4_blockscale.py)

## Performance and validation

Check packing order, scale exponent encoding, tail groups and dequantized reference values. Include conversion/prepacking cost when it occurs per invocation.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](gluon.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
