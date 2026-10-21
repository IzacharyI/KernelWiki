---
title: "Attention indexer: scoring, selection and paged-cache variants"
kind: implementation_knowledge
operator: attention_indexer
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scoring and selection in the attention indexer

The indexer selects the KV tokens or blocks that sparse attention will consume. Evaluate the complete score-to-selection path alongside the downstream attention kernel: a faster score kernel can lose its benefit through expensive selection, cache conversion or rank merging.

## FP8 MQA logits and shape-dependent scheduling

The [FlyDSL API](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fp8_mqa_logits_kernels.py) exposes `flydsl_fp8_mqa_logits`, a kernel-variant registry and a compilation cache. Its gfx942/gfx950 selection depends on query length, KV length and head count. Head-count divisibility must also match the selected MFMA atom; a power-of-two head count alone does not guarantee support.

The wrapper pads query rows, aligns output columns and can split KV columns across `grid.y` to fill the GPU for small query batches. That split produces separate score columns; it is not the partial-softmax reduction used by split-KV attention. Compare the active variant and full output extent before transferring a launch configuration between decode and prefill.

## Paged and low-bit variants

The [FlyDSL source references](flydsl_aiter.md) include contiguous FP8 scoring, paged FP8 scoring, FP4 paged scoring and a prefill variant. The [MXFP4 launcher in the Triton namespace](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_mqa_logits_mxfp4.py) imports gfx950 Gluon kernels and defines scale-layout, page-tile and scheduler helpers. Its kernel tile is bounded by page size so it does not span pages.

Keep payload packing, scale order, candidate-block size and page-table semantics with the scoring implementation. These details belong to the indexer contract, rather than separate quantization or gather operator entries.

## Selection across context-parallel ranks

The [FlyDSL DCP top-k merge](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/dcp_topk_merge.py) consumes gathered scores and local indices to produce selected KV indices. Upstream requires invalid candidates to have `-inf` scores; zero padding can outrank valid candidates and silently under-emit indices. Warm each shape before graph capture because the first invocation compiles the launcher.

Validate scores and selected indices together, including masked positions, short contexts, ties, padded candidates and rank ownership. Benchmark scoring, selection and sparse attention both separately and as a chain.

## Validation routes

- [FlyDSL FP8 MQA tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_mqa_logits.py).
- [FlyDSL paged FP4 tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4.py).
- [DCP merge tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_dcp_topk_merge.py).

These are source-derived implementation notes, not new GPU measurements or a backend ranking.
