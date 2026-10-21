# Gluon — shared memory / LDS layouts

## CDNA (gfx942/gfx950)

- LDS declared via TTGIR shared layout; capacity per CU: 64 KiB (gfx942), 160 KiB
  (gfx950); alloc granularity 512 B vs **1280 B** on gfx950 (`../hardware/planning-constants.md`).
- Swizzle: `PaddedSharedLayout`, blocked layouts recovered from TTGIR transcription.
- Pipeline stages = separate LDS buffers per stage (hand-sized; no `num_stages` knob).

## gfx1250

- `PartitionedSharedLayout` for WMMA operand staging (wave32).
- 320 KiB LDS partition model — probe occupancy per kernel.

## vs CuTeDSL smem/TMEM

| CuTeDSL | Gluon |
| --- | --- |
| TMEM accumulator (sm_100) | **scoped ceiling** — AGPR/VGPR on AMD |
| TMA swizzle modes | LDS swizzle + buffer resource |
| `make_swizzle` XOR | padded/blocked shared layouts |

## Footguns

- gfx950 **1280 B** LDS alignment waste on small tiles.
- gfx942 kernel on gfx950 without relayout → wrong results or spill.

## Anchors

- [triton-lang/triton `gluon/language/amd/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd)

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/expert_skills/skills/gluon_authoring/references/gluon/smem-lds-reference.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
