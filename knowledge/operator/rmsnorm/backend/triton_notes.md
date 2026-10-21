---
title: "AITER Triton RMSNorm: grid and row-width dispatch"
kind: implementation_notes
operator: rmsnorm
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_review
---

# AITER Triton RMSNorm: grid and row-width dispatch

[Runnable source bundle](../../../../kernels/rmsnorm/triton/aiter_row_tiled/README.md): `rmsnorm/triton/aiter_row_tiled`. The kernel body and naming helper are preserved from the pinned source; the standalone wrapper and tests are included. GPU verification of the extracted bundle is pending.

The current wrapper has separate policies for generic rows, very wide rows and large-M/small-N workloads. A single “persistent one-workgroup-per-CU” description no longer covers all forward calls.

## How the implementation selects work

`num_programs()` caps the generic grid at `min(M, num_sms)`. On gfx950, `num_programs_fwd()` instead permits four workgroups per CU when the selected block width is at most 8192. Wider blocks use the capped grid. This is a forward-only choice; backward uses its own path.

`block_size()` starts with the next power of two of N. The generic capacity bound is `65536 / element_size`. On gfx950, when a row exceeds that capacity, the blocked path uses a block width of 8192; the source comment associates this choice with avoiding register spills. This does **not** mean that every row wider than 8192 is automatically blocked.

The large-M/small-N branch has different forward and backward thresholds:

| Direction | Branch condition | Work decomposition |
| --- | --- | --- |
| Forward | `M > 8192` and `N <= 2048` | Multiple rows per program; `ceil(M / BLOCK_M)` programs |
| Backward | `M > 8192` and `N <= 1024` | Tiled row work plus partial weight-gradient reduction |

The forward tiled path uses `BLOCK_N = next_power_of_2(N)` and `BLOCK_M = max(8, min(16384 / BLOCK_N, 32))`. Warp/stage choices come from `get_normalization_config("rmsnorm_large_m_small_n", get_arch())`. This matters for per-head Q/K normalization: many short rows can serialize badly under a CU-capped grid.

## Performance evidence

The upstream `_should_use_large_m_small_n` docstring reports the following ratios for **MI308X, M=16384, bf16**, motivating its branch thresholds:

| N | Forward ratio | Backward ratio |
| ---: | ---: | ---: |
| 128 | 13.6× | 5.1× |
| 512 | 7.8× | 3.0× |
| 1024 | 5.0× | 1.2× |
| 1280 | 2.6× | approximately 1× |

These are **upstream source-reported ratios**, not KernelWiki measurements. The comment does not provide raw latency samples, a complete software stack or timing protocol. Use them as motivation to test that regime, not as a cross-backend ranking or a guaranteed improvement on another GPU.

## What to check when optimizing

- Confirm which branch the public wrapper actually selects before changing a grid or tile.
- Include any `.contiguous()` copies performed by the wrapper when they occur on the live path.
- Check normalized output, saved inverse RMS and both gradients when applicable.
- Re-test around M=8192 and the forward/backward N thresholds. Backward reduction cost changes the crossover.

## Sources

- [Wrapper, branch thresholds and reported ratios](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py)
- [Triton implementation](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py)
- [Related source and test references](triton_aiter.md)
