---
title: "Allreduce: AITER flydsl"
kind: implementation_knowledge
operator: allreduce
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Allreduce — AITER flydsl

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/quick_allreduce_int4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4.py) · **Recent**

Source documentation: Host launch for gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce.

Symbols: [`_cuda_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4.py#L35), [`_validate_ipc_process_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4.py#L47), [`_StEngine`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4.py#L70), [`QuickAllReduceInt4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4.py#L169).

### `aiter/ops/flydsl/quick_allreduce_int4_ipc.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4_ipc.py) · **Recent**

Source documentation: Uncached HIP IPC helpers for the FlyDSL INT4 quick-allreduce inbox.

Symbols: [`UncachedIpcHeap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/quick_allreduce_int4_ipc.py#L12).

### `aiter/ops/flydsl/kernels/quick_allreduce_int4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py) · **Recent**

Source documentation: gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce. INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host uses ST=1 when num_tiles ≤ the occupancy-clamped persistent grid. Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns atoms / world_size atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS stays atoms * 1152.

Symbols: [`clamp_grid_cap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L49), [`_f16x2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L73), [`_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L77), [`_splat_f16x2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L81), [`_clamp_fp16_overflow`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L85), [`_shuffle_f16x2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L102), [`_pair_signed_ext_f16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L106), [`_atom_bf16_to_f16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L123), [`_atom_f16_to_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L127), [`_f32_to_e4m3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L131), [`_e4m3_to_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L160), [`_pack_e4m3_word`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L171), [`_e4m3_decoding_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L182), [`_quant_atom_fp16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L186), [`_codec_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L197), [`_codec_dequant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L208), [`_i32_to_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L229), [`_store_v4i32_nt_global`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L233), [`_load_i32_nt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L247), [`_load_i32_uncached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L255), [`_invalidate_l1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L263), [`PackStorage`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L269), [`PackNormStorage`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L278), [`latency_max_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L286), [`rmsnorm_hidden_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L291), [`mxfp4_scale_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L301), [`make_quick_allreduce_int4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/quick_allreduce_int4.py#L306).

Parameters referenced in this file: `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/gfx1250_poc/test_gfx1250_allreduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/gfx1250_poc/test_gfx1250_allreduce.py)
- [op_tests/multigpu_tests/test_custom_allreduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce.py)
- [op_tests/multigpu_tests/test_custom_allreduce_capture_registration.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce_capture_registration.py)
- [op_tests/multigpu_tests/test_custom_allreduce_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce_fp8.py)
- [op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py)
- [op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4_rmsnorm.py)
- [op_tests/multigpu_tests/test_quick_all_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce.py)
- [op_tests/multigpu_tests/test_quick_all_reduce_gfx1250.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce_gfx1250.py)
- [op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
