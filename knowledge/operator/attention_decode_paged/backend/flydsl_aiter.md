---
title: "Paged and unified attention decode: AITER flydsl"
kind: implementation_knowledge
operator: attention_decode_paged
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Paged and unified attention decode — AITER flydsl

Paged attention must preserve the block table, sequence lengths, causal mask, query-head to KV-head mapping and cache dtype. Split-KV exposes parallelism at the cost of partial buffers and a reduction pass.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/pa_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode.py) · **Recent**

Source documentation: FP8 or BF16 paged-attention decode on gfx942/gfx950/gfx1250. Cache layouts are logical, not preshuffled. K/V use E4M3 FNUZ on gfx942 and OCP on gfx950/gfx1250. See kernels.pa_decode_kernel for Q/P quantization and MFMA/WMMA specialization details. Native BF16 uses unscaled, 8-element packed K/V caches. gfx1250 also accepts FP16 queries and plain BF16 V caches.

Symbols: [`_flydsl_pointer_dtype`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode.py#L30), [`launch_pa_decode_ps_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode.py#L41), [`pa_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode.py#L173).

### `aiter/ops/flydsl/pa_decode_tuning.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py) · **Recent**

Source documentation: Native FlyDSL autotuning and cached work-plan preparation for PA decode. Run with: python -m aiter.ops.flydsl.pa_decode_tuning --help

Symbols: [`_tuning_cache_tag`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L33), [`_cache_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L40), [`_PADecodeAutotuneResources`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L49), [`_pa_decode_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L153), [`_pa_decode_default`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L169), [`_validate_pa_decode_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L174), [`_pa_decode_autotuner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L208), [`_get_pa_decode_autotuner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L232), [`prepare_pa_decode_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L247), [`_positive_int`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L389), [`_parse_args`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L396), [`_quantize_kv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L468), [`_make_tuning_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L482), [`main`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/pa_decode_tuning.py#L548).

### `aiter/ops/flydsl/kernels/pa_decode/context.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/context.py) · **Recent**

Source documentation: Per-CTA operands, thread ownership and partition bounds for PA decode. Initialization stays explicit so the pipeline controls the order of global loads. Loop-carried softmax and output values are passed as SSA values, not stored here.

Symbols: [`PaDecodeContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/context.py#L16).

### `aiter/ops/flydsl/kernels/pa_decode/op_epilog.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_epilog.py) · **Recent**

Source documentation: Normalize PA output and store packed partition partials.

Symbols: [`PaDecodeEpilogue`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_epilog.py#L15).

### `aiter/ops/flydsl/kernels/pa_decode/op_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_gemm.py) · **Recent**

Source documentation: FP8/BF16 operand conversion and QK/PV MFMA/WMMA operations.

Symbols: [`PaDecodeGemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_gemm.py#L15), [`Gfx1250Gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_gemm.py#L103).

### `aiter/ops/flydsl/kernels/pa_decode/op_lds.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_lds.py) · **Recent**

Source documentation: LDS views, query staging and paged K/V/scale loads.

Symbols: [`PaDecodeLds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_lds.py#L20), [`PaDecodeQueryLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_lds.py#L76), [`PaDecodeKVLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_lds.py#L247), [`Gfx1250Memory`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_lds.py#L741).

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`.

### `aiter/ops/flydsl/kernels/pa_decode/op_softmax.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_softmax.py) · **Recent**

Source documentation: PA score masking, online softmax, and BF16/FP8 probability staging.

Symbols: [`PaDecodeSoftmax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/op_softmax.py#L15).

### `aiter/ops/flydsl/kernels/pa_decode/pipeline.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/pipeline.py) · **Recent**

Source documentation: PA decode prologue and single-/multi-query tile pipelines. The pipeline owns instruction ordering. Operation helpers keep descriptors and compile-time policy; loop-carried K/V, maxima, denominators and outputs remain explicit SSA values. Every LDS publication/retirement barrier stays here.

Symbols: [`PaDecodePipeline`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/pipeline.py#L25), [`Gfx1250Bf16Pipeline`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/pipeline.py#L618), [`Gfx1250Pipeline`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/pipeline.py#L844).

### `aiter/ops/flydsl/kernels/pa_decode/traits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py) · **Recent**

Source documentation: Compile-time specialization and storage layout for paged-attention decode.

Symbols: [`d256_pipe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L28), [`max_tiles_per_task`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L51), [`stage_page_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L63), [`_bf16_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L89), [`PaDecodeSchedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L99), [`PaDecodeTraits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L359), [`Gfx1250Traits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode/traits.py#L757).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/pa_decode_kernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_kernel.py) · **Recent**

Source documentation: FP8 and BF16 paged-attention tile kernel. K/V use e4m3 (FNUZ on gfx942, OCP on gfx950/gfx1250); BF16/FP16 Q and probabilities P are quantized to FP8. Q/key scales fold into QK, value scale and 1/FP8_MAX into the epilogue; softmax max/sum stay f32. Tuned gfx950 BF16 per-token MTP3/MTP4 and the head-256 page-16/64 pipe use K128 MFMA instead of K32, preserving normalized Q/P and operand layouts.

Symbols: [`compile_pa_decode_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_kernel.py#L85), [`compile_gfx1250_pa_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_kernel.py#L398).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/pa_decode_plan.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_plan.py) · **Recent**

Source documentation: GPU work planning for FlyDSL paged attention with mixed context lengths.

Symbols: [`compile_pa_decode_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_plan.py#L21), [`PADecodePlan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_plan.py#L149), [`plan_pa_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_plan.py#L196).

### `aiter/ops/flydsl/kernels/pa_decode_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_reduce.py) · **Recent**

Source documentation: Partitioned-softmax reduction kernel for FlyDSL paged attention.

Symbols: [`_shuffle_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_reduce.py#L30), [`_fast_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_reduce.py#L35), [`compile_pa_decode_ps_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/pa_decode_reduce.py#L41).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/cpp/test_pa_decode_gluon_aot.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/test_pa_decode_gluon_aot.py)
- [op_tests/op_benchmarks/triton/bench_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_decode.py)
- [op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py)
- [op_tests/op_benchmarks/triton/bench_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_unified_attention.py)
- [op_tests/test_flydsl_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_decode.py)
- [op_tests/test_pa_decode_bf16_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_decode_bf16_asm.py)
- [op_tests/triton_tests/attention/test_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode.py)
- [op_tests/triton_tests/attention/test_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_paged_attention_output_gate.py)
- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention.py)
- [op_tests/triton_tests/test_pa_decode_gluon.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_pa_decode_gluon.py)

## Performance and validation

Record page size, KV length distribution, head dimensions, number of query tokens and split count. Time the full split/reduce path; verify multi-token decode causality.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
