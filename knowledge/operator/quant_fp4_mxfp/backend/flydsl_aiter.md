---
title: "Packed low-bit and microscaling formats: AITER flydsl"
kind: implementation_knowledge
operator: quant_fp4_mxfp
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Packed low-bit and microscaling formats — AITER flydsl

Low-bit values are packed and interpreted together with scale metadata. Logical tensor sizes differ from storage sizes, and consumers may require shuffled scale/weight layouts.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/mxfp4_kname.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py) · **Recent**

Source documentation: The A-scale layout GEMM1 must emit for a block_m and output dtype. This is a GEMM1/GEMM2 contract, not a tuning knob: BM16 FP4 output writes the native scale layout and the matching GEMM2 reads it back. FP8 output uses the regular scale layout. Every caller must agree, so the rule lives here.

Symbols: [`native_scale_layout_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L45), [`_tokenize_mxfp4_kname`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L64), [`_parse_mxfp4_g1_kname`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L99), [`_parse_mxfp4_g2_kname`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L124), [`_is_mxfp4_kname`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L150), [`parse_flydsl_v2_gemm2_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L156), [`parse_g2_kname_any`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_kname.py#L176).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py) · **Recent**

Source documentation: Grouped contiguous-M A8W4 preshuffle MoE GEMM for gfx1250 (TDM pipeline).

Symbols: [`is_fx_set_register_available`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py#L51), [`_pack_bytes_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py#L106), [`launch_gemm_a8w4_tdm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py#L125).

Parameters referenced in this file: `num_buffers`, `num_warps`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

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

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
