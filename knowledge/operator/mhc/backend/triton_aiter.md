---
title: "Manifold-constrained hyper-connections: AITER triton"
kind: implementation_knowledge
operator: mhc
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Manifold-constrained hyper-connections — AITER triton

MHC paths implement pre/post residual mixing and normalization-related stages, including Sinkhorn-style transforms in selected entries. Fusion changes intermediate storage and reduction order.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/mhc.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py) · **Recent**

Source documentation: Fused mHC layer in one Triton launch (or a split-K + reduce-apply launch pair when NUM_KSPLIT > 1).

Symbols: [`_validate_dot_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L44), [`mhc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L63), [`mhc_post`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L381), [`_gluon_ra_norm_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L526), [`_mhc_post_pre_gluon_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L541), [`_gluon_post_pre_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L597), [`_gluon_reduce_apply_layouts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L614), [`_gluon_split_num_stages`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L622), [`_mhc_post_pre_gluon_split_k`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L641), [`mhc_post_pre_gfx1250_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L666), [`_mhc_post_pre_gluon_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L698), [`mhc_post_pre`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L904), [`_mhc_pre_dsv4_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1361), [`_mhc_head_dsv4_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1401), [`_validate_dsv4_parameters`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1420), [`_mhc_pre_dsv4_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1468), [`_MHCPreDSV4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1528), [`mhc_pre_dsv4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1574), [`_MHCPostDSV4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1611), [`mhc_post_dsv4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1634), [`_mhc_head_dsv4_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1652), [`_MHCHeadDSV4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1709), [`mhc_head_dsv4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc.py#L1725).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_C_SPLIT`, `BLOCK_K`, `BLOCK_M`, `BLOCK_M_POST_RES`, `BLOCK_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py) · **Recent**

Source documentation: Fused delayed ("shifted") mHC seam used by DeepSeek-V4.1: post-mix, gate projection, pre/post/comb gates, stream collapse with the carried pre gate and its RMSNorm in two Triton launches. See the kernel module for the math and structure.

Symbols: [`mhc_fused_post_pre_delayed_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py#L20).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_M`, `NUM_KSPLIT`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/fusions/mhc.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py) · **Recent**

Source documentation: Triton kernel for mHC (manifold-constrained Hyper Connection) operations.

Symbols: [`_mhc_apply_pre_mix_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L13), [`_mhc_asymmetric_sinkhorn_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L59), [`_mhc_head_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L108), [`_mhc_fused_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L202), [`_mhc_fused_split_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L419), [`_mhc_reduce_apply_res_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L511), [`_mhc_reduce_apply_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L598), [`_mhc_post_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L855), [`_mhc_post_pre_split_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L982), [`_mhc_post_pre_reduce_apply_res_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L1165), [`_mhc_post_pre_reduce_apply_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc.py#L1279).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_K`, `BLOCK_M`, `BLOCK_M_POST_RES`, `BLOCK_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py) · **Recent**

Source documentation: Triton kernels for the delayed ("shifted") mHC seam of DeepSeek-V4.1.

Symbols: [`_split4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py#L38), [`_mhc_fused_post_pre_delayed_rmsnorm_main_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py#L53), [`_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py#L230).

Parameters referenced in this file: `BLOCK_C`, `BLOCK_M`, `NUM_KSPLIT`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx942/triton/fusions/mhc_dsv4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/fusions/mhc_dsv4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=1024.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=1024.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=32768.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=32768.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=512.json)
- [aiter/ops/triton/configs/gfx942/triton/mhc/mhc_post/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/mhc/mhc_post/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/mhc_dsv4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/mhc_dsv4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=1024.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=1024.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=32768.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=32768.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=512.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/mhc/mhc_post/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_post/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc.py)
- [op_tests/op_benchmarks/triton/bench_mhc_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/triton_tests/fusions/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc.py)
- [op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py)
- [op_tests/triton_tests/utils/mhc_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mhc_ref.py)
- [op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_mhc_post_only.py)
- [op_tests/test_flydsl_mega_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mega_mhc.py)
- [op_tests/test_mhc.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mhc.py)

## Performance and validation

Match channel count, mixing matrices, iteration settings and all residual outputs. Include surrounding pre/post work in a fused comparison.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
