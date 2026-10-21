---
title: "Manifold-constrained hyper-connections: AITER aiter"
kind: implementation_knowledge
operator: mhc
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Manifold-constrained hyper-connections — AITER aiter

MHC paths implement pre/post residual mixing and normalization-related stages, including Sinkhorn-style transforms in selected entries. Fusion changes intermediate storage and reduction order.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/mhc.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py) · **Recent**

Source documentation: k elements per hi/lo interleave block in the packed BF16 fn layout. A lane's fn fragment must cover a whole block, i.e. vec_tile = tile_k / (warp_size / 16) >= kb. wave32 reaches 16 at tile_k=32 and its WMMA path consumes a whole 16-float block as two verbatim fragments; wave64 only reaches 8 there, so it uses 8-element blocks and keeps tile_k=32. MUST stay in lockstep with mhc_fn_kb in csrc/kernels/mhc_kernels.cu.

Symbols: [`mhc_pre_gemm_sqrsum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L18), [`mhc_fn_kb`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L28), [`mhc_shuffle_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L43), [`mhc_pre_big_fuse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L77), [`mhc_pre_big_fuse_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L96), [`_validate_mhc_res_input`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L130), [`_get_compiled_mhc_res_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L144), [`_run_mhc_res_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L152), [`mhc_res_repeat_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L180), [`mhc_res_shuffle_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L204), [`mhc_res_unshuffle_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L213), [`mhc_res_shuffle_enabled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L229), [`_mhc_res_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L247), [`_check_mhc_post_pre_backend`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L259), [`mhc_res_ks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L266), [`_check_mhc_res_preshuffle_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L282), [`mhc_res_layout_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L290), [`mhc_res_shuffle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L295), [`mhc_res_repeat_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L304), [`mhc_res_repeat`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L320), [`mhc_res_unshuffle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L347), [`get_mhc_pre_splitk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L355), [`_mhc_fused_valid_splitk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L410), [`_mhc_fused_fill_splitk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L419), [`_mhc_fused_config_gfx950_256`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L429), [`_mhc_fused_config_gfx942_80`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L455), [`_mhc_fused_config_gfx1250_256`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L478), [`_mhc_fused_bf16_shuffled_config_gfx1250_256`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L527), [`_mhc_fused_config_default`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L555), [`get_mhc_fused_post_pre_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L607), [`mhc_pre_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L647), [`mhc_pre`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L674), [`_mhc_post`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L771), [`mhc_post`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L782), [`get_mhc_pre_splitk_large_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L796), [`mhc_fused_post_pre_gemm_sqrsum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L806), [`mhc_fused_post_pre_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L823), [`mhc_fused_post_pre_large_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L855), [`_mhc_fused_post_pre_gluon`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L922), [`mhc_fused_post_pre`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mhc.py#L991).

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py)
- [op_tests/test_flydsl_mega_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mega_mhc.py)
- [op_tests/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mhc.py)
- [op_tests/op_benchmarks/triton/bench_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc.py)
- [op_tests/op_benchmarks/triton/bench_mhc_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/triton_tests/fusions/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc.py)
- [op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/triton_tests/utils/mhc_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mhc_ref.py)

## Performance and validation

Match channel count, mixing matrices, iteration settings and all residual outputs. Include surrounding pre/post work in a fused comparison.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
