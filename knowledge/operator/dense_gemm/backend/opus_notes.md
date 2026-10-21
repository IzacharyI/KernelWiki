---
title: "AITER OPUS: exact kernel ids and shape-driven dispatch"
kind: implementation_notes
operator: dense_gemm
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_review
---

# AITER OPUS: exact kernel ids and shape-driven dispatch

`opus_gemm` and `opus_bmm` accept a caller-selected final `kid`. They validate and execute that instance; they do not choose a faster instance from the shape. `gemm_a16w16_opus` is a separate compatibility entry that performs OPUS-only tuned lookup or a per-architecture heuristic when no explicit kernel id is supplied.

## Public contracts

| Entry | Logical input shape | Selection |
| --- | --- | --- |
| `opus_gemm` | X `[M,K]`, W `[N,K]`, caller-owned Y `[M,N]` | Mandatory exact `kid` |
| `opus_bmm` | X `[B,M,K]`, W `[B,N,K]`, Y `[B,M,N]` | Mandatory exact `kid`; family must support BMM |
| `gemm_a16w16_opus` | A16W16 GEMM compatibility API | Explicit id, OPUS tuning row, then OPUS heuristic |

Inputs are K-contiguous and Y is N-contiguous. Family-specific scale, bias, workspace, output dtype and tile restrictions still apply. An A8 GEMM's internal three-dimensional ABI does not make it a public BMM implementation.

The upstream documentation distinguishes plain A8W8, blockscale and preshuffled-blockscale GEMM families from MXFP8 BMM. Some architecture tables are intentionally empty. Validate a `kid` against its instance architecture and family rather than inferring capabilities from its integer value.

## Benchmark implications

Benchmarking a hand-selected exact `kid` answers a different question from benchmarking `aiter.gemm_a16w16`, which goes through the global multi-backend dispatcher. A useful result records both the explicit instance and the public dispatch behavior. Include split-K workspace/reduction cost and any required weight preparation.

There is no new measured speedup in this note. The distinction prevents attributing a selection-policy improvement to kernel arithmetic or assuming a tuned candidate is already the production choice.

## Sources

- [OPUS API, current families and dispatch contracts](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/README.md)
- [Global tuned GEMM dispatcher](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/tuned_gemm.py)
- [OPUS source references](opus_aiter.md)
