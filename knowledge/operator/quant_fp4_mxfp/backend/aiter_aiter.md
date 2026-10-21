---
title: "Packed low-bit and microscaling formats: AITER aiter"
kind: implementation_knowledge
operator: quant_fp4_mxfp
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Packed low-bit and microscaling formats — AITER aiter

Low-bit values are packed and interpreted together with scale metadata. Logical tensor sizes differ from storage sizes, and consumers may require shuffled scale/weight layouts.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/iq2r_glm53.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py) · **Recent**

Source documentation: GLM-5.3 IQ2R MoE on the packed layout (TP4 and TP8, gfx950). The packed layout is a byte-exact relayout of the canonical IQ2R records, so decoded FP8 weights are unchanged: * codebook signs move into the record sign bytes (codebook bit 7 cleared); * the four 9-bit indices of an atom pack as low32 + high4; * gate/up record signs transpose into four 8-bit planes, and the gate/up records are regrouped into quads of four 16-column blocks.

Symbols: [`_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L69), [`_atoms`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L77), [`_normalize_codebook_signs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L103), [`_pack_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L123), [`_transpose_signs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L138), [`_quad_gate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L149), [`iq2r_glm53_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L185), [`iq2r_glm53_gate_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L209), [`iq2r_glm53_slice_gate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L216), [`IQ2RGlm53Config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L236), [`_default_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L244), [`_config_problem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L251), [`_tuned_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L285), [`iq2r_glm53_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L323), [`_task_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L349), [`IQ2RGlm53Workspace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L354), [`iq2r_glm53_moe_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/iq2r_glm53.py#L414).

### `aiter/ops/iq2r.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py) · **Recent**

Source documentation: Compiled gfx950 operations for native-basis IQ2R.

Symbols: [`_iq2r_encode_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L16), [`_iq2r_materialize_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L32), [`iq2r_glm53_m1_route_quant_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L47), [`iq2r_glm53_sort_quant_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L59), [`iq2r_glm53_prefill_sort_quant_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L76), [`iq2r_glm53_route_reduce_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L90), [`iq2r_glm53_gate_m1_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L99), [`iq2r_glm53_gate_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L112), [`iq2r_glm53_down_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L128), [`iq2r_glm53_down_route9_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L142), [`iq2r_glm53_down_reduce_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L155), [`_validate_gpu_weights`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L170), [`iq2r_encode_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L188), [`iq2r_materialize_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L260), [`iq2r_materialize_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r.py#L283).

### `aiter/ops/iq2r_encoder.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py) · **Recent**

Source documentation: Importance-aware reference encoder for AITER's native-basis IQ2R format.

Symbols: [`_pad_k`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L41), [`iq2r_scale_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L45), [`_scale_importance`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L60), [`iq2r_initial_codebook`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L64), [`iq2r_reserve_zero_codeword`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L74), [`_assign_codewords`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L87), [`_e8m0_candidates`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L104), [`iq2r_learn_codebook`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L116), [`iq2r_encode_reference`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_encoder.py#L169).

### `aiter/ops/iq2r_format.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py) · **Recent**

Source documentation: Storage layout of AITER's native-basis IQ2R weights. CPU-only, with no compiled extension, so checkpoint tooling can size, slice and validate IQ2R byte buffers before touching a GPU.

Symbols: [`_require_int`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L42), [`iq2r_k_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L49), [`iq2r_n_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L58), [`iq2r_physical_n_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L67), [`iq2r_padded_k`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L76), [`iq2r_data_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L80), [`iq2r_aux_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L90), [`iq2r_packed_sizes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L94), [`iq2r_storage_bits_per_weight`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L100), [`IQ2RMetadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L106), [`_validate_output_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L145), [`iq2r_slice_output_auxiliary`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L159), [`iq2r_slice_input_data`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L190), [`_validate_byte_matrix`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L225), [`iq2r_validate_expert_weights`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_format.py#L242).

### `aiter/ops/iq2r_reference.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py) · **Recent**

Source documentation: Pure-Torch correctness oracle for AITER IQ2R weights. The implementation follows the physical byte layout consumed by the gfx950 scaled-FP8 MFMA operand. It is deliberately explicit and optimized for audit clarity rather than serving speed.

Symbols: [`iq2r_physical_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L37), [`iq2r_tile_views`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L49), [`iq2r_metadata_word`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L88), [`iq2r_set_metadata_word`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L98), [`_materialize_one`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L108), [`iq2r_materialize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/iq2r_reference.py#L167).

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/configs/iq2r_glm53_tuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/iq2r_glm53_tuned.csv)
- [aiter/configs/iq2r_glm53_untuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/iq2r_glm53_untuned.csv)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py)
- [op_tests/test_iq2r_encoder.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_encoder.py)
- [op_tests/test_iq2r_format.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_format.py)
- [op_tests/test_iq2r_glm53.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm53.py)
- [op_tests/test_iq2r_glm5_calibrate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm5_calibrate.py)
- [op_tests/test_iq2r_glm5_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_glm5_compile.py)
- [op_tests/test_iq2r_hip.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_hip.py)
- [op_tests/test_iq2r_reference.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_iq2r_reference.py)
- [op_tests/test_quant_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4.py)
- [op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp4_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp4_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_quant_mxfp4_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_quant_mxfp4_fp8.py)
- [op_tests/triton_tests/quant/test_fused_mxfp4_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_fused_mxfp4_quant.py)
- [op_tests/triton_tests/quant/test_mxfp4_jit_import.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_mxfp4_jit_import.py)
- [op_tests/triton_tests/quant/test_quant_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_mxfp4.py)
- [op_tests/triton_tests/quant/test_quant_mxfp4_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/quant/test_quant_mxfp4_blockscale.py)

## Performance and validation

Check packing order, scale exponent encoding, tail groups and dequantized reference values. Include conversion/prepacking cost when it occurs per invocation.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
