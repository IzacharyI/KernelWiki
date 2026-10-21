---
title: "GQA and MQA in unified paged attention"
kind: implementation_knowledge
operator: gqa_mqa_attention
backend: triton
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# GQA and MQA in unified paged attention

[unified_attention](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/attention/unified_attention.py) computes `num_queries_per_kv = num_query_heads // num_kv_heads` and maps grouped query heads onto shared K/V heads in [the device implementation](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/_triton_kernels/attention/unified_attention.py). MQA is the single-KV-head case. This is a workload variant of the unified attention implementation, not a new kernel merely because it has another operator page.

## Scheduling variants

The wrapper sizes the query tile with the query-to-KV head ratio and derives `BLOCK_Q` from `BLOCK_M`. A launch configuration tuned for equal Q/KV head counts can therefore change its work decomposition under GQA. It chooses between 2-D attention and 3-D segmented attention plus a merge.

The Python namespace exposes both Triton and Gluon. An explicit backend argument or architecture-dependent default selects the path; gfx950 and gfx1250 have Gluon implementations with their own eligibility checks. The current public wrapper requires causal attention.

Validate grouped-head mapping, head dimensions, nonuniform context lengths and quantization scales. Benchmark prefill and decode separately, and include the segmented merge when selected. The shared KV bytes and changed query grouping are the reason this deserves a workload-specific page, rather than duplicating a standard MHA timing.

## Evidence and validation

- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/triton_tests/attention/test_unified_attention.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
