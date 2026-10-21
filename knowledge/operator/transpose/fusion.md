---
title: transpose — fusion (don't ship a standalone transpose)
kind: technique
operator: transpose
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
---

# transpose — fusion

## The thesis
A standalone transpose is **almost always an anti-pattern** on a tuned serving path: it costs a full HBM
round-trip (`2·bytes/5.3 TB/s`) for zero arithmetic. The win is to make the transpose **disappear** into a
neighbor. Spend effort deleting transposes, not speeding them up.

## Fusion targets (in priority order)
| where | how the transpose vanishes | link |
|---|---|---|
| **GEMM operand** | `transpose_b=true` folds Bᵀ into the **MFMA operand layout** — the matrix core consumes the transposed fragment directly from LDS; no separate transpose kernel. sglang `nn.Linear` is already TN. | [operators/dense_gemm/overview.md](../dense_gemm/README.md) |
| **weight pre-shuffle** | a one-time offline transpose-into-MFMA-layout is exactly [operators/layout_shuffle/overview.md](../layout_shuffle/README.md) (aiter `shuffle_weight` / `bpreshuffle`) — pays the move **once at load**, not per call. | [operators/layout_shuffle/overview.md](../layout_shuffle/README.md) |
| **consumer load (transpose-on-read)** | stage into LDS, read the transposed fragment with a swizzled `ds_read_b128` (gfx942) or `ds_read_*_tr_b16` (gfx950) — the transpose happens for free during the load the kernel already does. | [operators/transpose/tuning.md](tuning.md) §3,§5 |
| **attention reshape** | `[B,S,H]↔[B,H,S]` permutes for QKᵀ/PV are folded into the FMHA kernel's tile loads, not a separate `permute().contiguous()`. | [operators/attention_prefill_fmha/overview.md](../attention_prefill_fmha/README.md) |
| **+cast / +quant** | if a transpose is unavoidable, fuse the dtype cast (bf16→fp8) into it so the single HBM pass also quantizes. ⚠ the **cast** carries the FNUZ/OCP dialect risk, not the move. | [operators/quant_dequant_fp8/overview.md](../quant_dequant_fp8/README.md) |

## When a standalone transpose is justified
- A layout the consumer genuinely cannot ingest in its native form **and** that is reused enough to amortize
  the pass (then prefer pre-shuffling it **once** → [operators/layout_shuffle/overview.md](../layout_shuffle/README.md)).
- A debugging / reference path. In production, a top-N standalone `transpose`/`permute`+`contiguous` in a
  profile is a fusion-opportunity flag.

## Cross-links
[operators/transpose/tuning.md](tuning.md) · [operators/layout_shuffle/overview.md](../layout_shuffle/README.md) ·
[operators/dense_gemm/overview.md](../dense_gemm/README.md) · [languages/hip_cpp/lds_async.md](../../general/backend/hip_cpp/lds_async.md).

## Sources
- MFMA operand layout consumes transposed fragments from LDS (transpose folded into GEMM): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Transpose LDS staging / swizzle: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- Pre-shuffle = amortized one-time transpose: [operators/layout_shuffle/overview.md](../layout_shuffle/README.md) (aiter `shuffle_weight`).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/transpose/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
