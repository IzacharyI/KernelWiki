---
title: "Chunked prefill over an existing paged prefix"
kind: implementation_knowledge
operator: chunked_prefill
backend: triton
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Chunked prefill over an existing paged prefix

In [unified_attention](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/attention/unified_attention.py), each request has a current query length and a total used KV length. [the device implementation](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/_triton_kernels/attention/unified_attention.py) derives `context_len = seq_len - cur_batch_query_len`, so causal positions are offset by the existing prefix. This directly represents a prompt chunk attending to cached tokens and earlier tokens in the same chunk.

## Two useful entry routes

| Route | Contract to preserve |
| --- | --- |
| Unified attention | `cu_seqlens_q` describes the current query chunks; `seqused_k` and the block table describe the total visible KV. The selected 2-D/3-D path and backend determine temporary storage and merge work. |
| [context_attention_fwd](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/attention/pa_prefill.py) | Accepts current Q/K/V, paged K/V caches, block locations and per-request lengths; supports `skip_decode` for mixed batches. |

The latter wrapper has FP8-cache, ALiBi, GQA and sliding-window variants. Its [prefill device kernels](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/_triton_kernels/attention/pa_prefill.py) use different cache addressing from a dense contiguous attention interface.

Validate chunk boundaries against a full causal-attention reference, including an empty prefix, a partial final page, uneven chunk sizes and mixed decode requests. Report query-token count and cached-prefix length separately; two cases with the same total sequence length can have very different arithmetic and cache traffic.

## Evidence and validation

- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/triton_tests/attention/test_unified_attention.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/triton_tests/attention/test_pa_prefill.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
