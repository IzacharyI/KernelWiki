---
title: "OPUS MXFP4 indexer: architecture layouts and dynamic schedules"
kind: implementation_knowledge
operator: attention_indexer
backend: opus
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# OPUS MXFP4 indexer: architecture layouts and dynamic schedules

[The OPUS paged-MQA-logits API](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/opus/pa_mqa_logits_mxfp4.py) dispatches to different native bodies on gfx950 and gfx1250. These are index-scoring kernels for sparse attention; they do not perform the final probability-value attention product.

| Source | What to track |
| --- | --- |
| [gfx950 scorer](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_gfx950.cuh) | MFMA-permuted cache/scale layouts. |
| [gfx1250 scorer](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_gfx1250.cuh) | The architecture's natural input layout and instance contract. |
| [Device schedule builder](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_sched.cuh) | Tile enumeration, CTA scheduling, scratch records and split target. |

The schedule builder uses per-instance resident-CTA counts and rounds work toward complete occupancy rounds. Short and long tile lists use different builder widths/grids. Include the builder in latency when metadata is rebuilt per invocation; exclude it only when the real serving path reuses a valid schedule.

The upstream test uses natural 3-D layouts on gfx1250 and permuted 4-D layouts on gfx950, with an independent FlyDSL cross-check for gfx950 corner cases. It covers causal prefill, fresh/chunked compressed sparse attention and multi-token decode. Keep those cases separate when reporting a best configuration, and validate scores and downstream selected indices.

## Evidence and validation

- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_pa_mqa_logits_mxfp4_opus.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
