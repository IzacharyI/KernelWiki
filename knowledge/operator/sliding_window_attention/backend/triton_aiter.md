---
title: "Sliding-window attention: mask and tile-pruning variants"
kind: implementation_knowledge
operator: sliding_window_attention
backend: triton
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sliding-window attention: mask and tile-pruning variants

[unified_attention](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/attention/unified_attention.py) translates its left-window argument to `SLIDING_WINDOW = 1 + window_size[0]`. [the device implementation](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/_triton_kernels/attention/unified_attention.py) computes the first allowed key from each query's absolute position and prunes unreachable KV tiles before applying the elementwise mask. Preserve that inclusive/exclusive convention when moving between attention APIs.

## Scheduling and fusion boundaries

A window changes both visible KV traffic and the wrapper's 2-D/3-D selection. The [paged-prefill wrapper](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/triton/attention/pa_prefill.py) also exposes `sliding_window`, using `0` or `None` to disable it. Do not pass the same numeric argument through both interfaces without checking their conventions.

This page describes the Triton route; the shared wrapper can select architecture-specific Gluon kernels. Recheck their mask and dtype eligibility instead of assuming every optional feature is supported by every route.

Validate positions at the left boundary, prefix/chunk offsets, short sequences and partial pages. Time at several context lengths above and below the window size. A full-context baseline and a windowed kernel solve different contracts unless the baseline applies the same mask.

## Evidence and validation

- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/triton_tests/attention/test_unified_attention.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/triton_tests/attention/test_pa_prefill.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
