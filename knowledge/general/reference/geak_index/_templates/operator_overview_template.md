---
title: <operator> — overview
kind: operator_overview
operator: <operator_id>
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: YYYY-MM-DD
sources: []
---

# <operator>

## TL;DR
> What it is in one sentence + the single most important optimization fact.

## Math contract
- exact definition / shapes / layouts (transpose, bias, epilogue), dtype in/out/accumulate.

## Shape regimes
- prefill vs decode (M/batch ranges), the shapes that dominate GPU time for typical LLMs.

## Where it matters (Amdahl)
- typical %GPU-time on real models; what speedup is needed to move e2e.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢/🟡/⚪ | `backend/triton.md` |
| flydsl | … | `backend/flydsl.md` |
| hip | … | `backend/hip.md` |
| ck | … | `backend/ck.md` |
| asm | … | `backend/asm.md` |
| tilelang | … | `backend/tilelang.md` |
| aiter | … | `backend/aiter.md` |
| hipblaslt | … | `backend/hipblaslt.md` |
| … | | |

## Fusion neighbors
- ops it can fuse with (bias/act/residual/quant epilogue; norm+quant; etc.) → see `fusion.md`.

## Numerics
- accuracy considerations → see `numerics.md`.

## How to bench
- canonical isolated benchmark + oracle (so every backend card compares apples-to-apples).

## Sources
- <primary sources>

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/index/_templates/operator_overview_template.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
