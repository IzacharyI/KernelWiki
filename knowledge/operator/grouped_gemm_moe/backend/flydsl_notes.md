---
title: "AITER FlyDSL MoE: requested tiles versus executed tiles"
kind: implementation_notes
operator: grouped_gemm_moe
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_review
---

# AITER FlyDSL MoE: requested tiles versus executed tiles

The MoE wrapper resolves tile sizes, persistence and stage-2 reduction mode before compilation/launch. A tuning record must describe the resolved execution, because requested parameters can be changed by these rules.

## Dimension-dependent tiles

`pick_flydsl_stage1_tile_n(inter_dim)` and `pick_flydsl_stage2_tile_k(inter_dim)` return 256 when the intermediate dimension is divisible by 256, otherwise 128. The wrapper emits a deduplicated warning when it overrides an incompatible tile. For example, an intermediate dimension of 640 cannot use the 256 tile on these axes even if a tuning row requests it.

The kernel name records stage, A/B/output types, M/N/K tiles, optional mode and a separate sorting block size. `_sbm` is present when sorting uses a positive block size different from the compute M tile. Preserve that distinction when pairing sorted token metadata with stage 1 and stage 2.

## Persistence and reduction are correctness constraints too

HIP limits grid.y/grid.z to 65535. `resolve_flydsl_grid_y_persist_m()` increases persistence to at least `ceil(num_m_blocks / 65535)`, in addition to respecting the requested value. Very large token/expert workloads can therefore execute a different persistent schedule from the requested one.

`requires_flydsl_stage2_reduce()` detects outputs whose byte extent exceeds `0xFFFFFFFF`. Stage-2 mode resolution also considers explicit force-reduce and FP8 stage-2 settings. The atomic and reduction paths can have different temporary-storage, initialization and numerical behavior; time the selected complete path.

## How to evaluate a candidate

Record `inter_dim`, token count, model dimension, data types, sorting block size, requested/resolved tiles and stage-2 mode. Confirm that the compiled kernel name and dispatch match the intended configuration. Check expert routing, padding, gating and reduction outputs, then compare complete two-stage latency including sorting or activation steps that remain on the live path.

No new GPU performance measurement is attached to this note. It explains source constraints that prevent misleading tuning comparisons.

## Sources

- [MoE wrapper, naming and resolution rules](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/moe_kernels.py)
- [AITER fused-MoE dispatch](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fused_moe.py)
- [Implementation and test references](flydsl_aiter.md)
