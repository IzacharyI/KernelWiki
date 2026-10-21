---
title: "Reduction: AITER asm"
kind: implementation_knowledge
operator: reduction
backend: asm
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Reduction — AITER asm

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

AITER ships architecture-specific code objects and dispatch metadata. Binary family names identify candidates, but do not prove supported shapes or performance; follow the wrapper and metadata that select an object.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

## Assembly code-object families

These are shipped code objects and their metadata, grouped by family rather than treating each compiled tile as a new operator. This records provenance and selection material; assembly internals have not been reconstructed from binaries.

| Architecture / family | Files | Changed in the period |
| --- | ---: | ---: |
| [hsa/gfx950/reduce](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/hsa/gfx950/reduce) | 4 | 4 |

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
