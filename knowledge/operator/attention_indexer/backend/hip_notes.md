---
title: "Sparse block selection: scoring, top-k and fused CP selection"
kind: implementation_knowledge
operator: attention_indexer
backend: hip
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse block selection: scoring, top-k and fused CP selection

The [native sparse-selector contract](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/sparse_attn/msa/pa_sparse_block_select.hpp) combines score generation with sparse-page-table construction. Its constants define a specialized family: index head dimension 128, block size 128, top-k 16 and eight pages per selected block. These are not defaults for arbitrary top-k workloads.

## Native variants

[The device kernels](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/sparse_attn/msa/pa_sparse_block_select_kernels.cuh) implement decode and prefill score/selection stages. [The top-k translation unit](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/sparse_attn/msa/pa_sparse_block_topk.cu) instantiates the supported top-k variants through the shared table; each generated entry is a specialization of this selector, not another operator.

The context-parallel selector is a distinct fused path: it nominates local shard candidates, publishes through IPC, merges peer candidates and emits the sparse page table. Its output must agree across ranks. Keep the initial/local block policy, valid-page counts and query-to-request mapping consistent with the downstream sparse attention kernel.

Compare complete score-to-page-table cost for decode and prefill, and measure CP communication separately from local score generation. Verify selected indices and sparse context lengths as well as the score tensor, especially around partially filled blocks and short requests.

## Evidence and validation

- [op_tests/test_msa_block_select.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_msa_block_select.py)
- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/multigpu_tests/test_msa_block_select_cp.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
