---
title: "AITER FlyDSL A16W16: useful tiles and occupancy-aware policy"
kind: implementation_notes
operator: dense_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_review
---

# AITER FlyDSL A16W16: useful tiles and occupancy-aware policy

`GemmConfigPruner` filters candidate policies using both padded-work efficiency and estimated occupancy. This makes its candidate set depend on M/N/K and device properties, before any measured comparison happens.

The useful-work ratio divides logical `M*N*K` by padded M/N/K work, including split-K. The occupancy estimate combines waves per workgroup, thread limits, LDS capacity and the total grid. LDS demand takes the larger of the staged operand tiles and the K-wave reduction requirement.

The policy caps split-K growth when the M/N tile grid already fills the CUs, bounds tile-grid expansion, and restricts nonzero `group_m` to sufficiently large grids with a multiple-of-eight tile count. It retains an additional `k_waves` choice only when estimated occupancy improves over the otherwise equivalent policy.

These are source-level heuristics. They can explain why a tile never reaches a benchmark, but do not establish that an excluded tile is slow on every workload. When analyzing a result, record the generated candidates, the surviving policies and the measured winning policy separately.

The source contains a distinct gfx1250 tile set and gfx950 validation logic. Use the implementation for the target architecture rather than transferring a tile list based only on dtype. No new GPU timings were collected for this note.

## Sources

- [Policy generation and pruning](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py)
- [Implementation and test references](flydsl_aiter.md)
