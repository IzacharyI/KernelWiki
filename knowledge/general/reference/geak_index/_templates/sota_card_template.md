---
title: <operator> on <backend> — SOTA card
kind: sota_card
operator: <operator_id>
backend: <backend_id>
gens: [gfx942]
dtypes: [bf16]
regimes: [prefill]
status: sota | competitive | experimental | legacy | na
updated: YYYY-MM-DD
sources: []
---

# <operator> × <backend>

## TL;DR (one-line decision)
> When this `operator × backend` is the right choice (and when it is NOT). If `status: na`, state the
> reason here and stop (e.g. "no Python rebind seam for this library GEMM → authored kernel can't be wired").

## SOTA implementation(s)
The *best known* implementation(s) for this cell. Enumerate all credible candidates; recommend one.

| impl | source (`repo@commit:path` / blog / paper) | gens / dtypes / shapes | measured perf (`value @ hw, ROCm/lib ver, date`) | when it's best |
|---|---|---|---|---|
| <name> | <src> | <support> | <measured> | <regime/shape> |

## Config space / knobs (backend-specific)
- key tunables + recommended ranges (e.g. for triton: `BLOCK_M/N/K`, `matrix_instr_nonkdim=16`,
  `waves_per_eu`, `num_stages`, `num_warps`, `SPLIT_K`; for flydsl: `tile_m/n/k`, `split_k`,
  `block_m/n_warps`, `waves_per_eu`, `b_preshuffle`, `stages`).

## Numerics / parity
- dtype, accumulation, tolerance vs reference; any argmax/tie-break risk; quant accuracy gate if applicable.

## Integration (how it gets used at serving time)
- rebind seam / call site (e.g. `aiter.tuned_gemm:gemm_a16w16`), env/flags, overlay method;
  how to deploy without editing site-packages; **how to verify it actually engages** (log marker).

## Pitfalls & anti-patterns
- known failure modes, shape/bias mismatches, host fork-storms, varlen instability, etc.

## How to verify (bench + oracle)
- exact bench command + correctness oracle + the gate (delta > band AND non-overlap AND engaged).

## Alternatives / cross-links
- other backends for this operator: `./triton.md`, `./ck.md`, …

## Sources
- <primary sources, per sourcing_rules.md>

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/index/_templates/sota_card_template.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
