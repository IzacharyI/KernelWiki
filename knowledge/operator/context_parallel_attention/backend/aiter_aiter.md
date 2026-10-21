---
title: "Round-robin context-parallel MLA and index selection"
kind: implementation_knowledge
operator: context_parallel_attention
backend: aiter
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Round-robin context-parallel MLA and index selection

[The MLA wrapper](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/mla.py) and [native attention bindings](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/attention.py) expose `cp_world_size`, `cp_rank` and global KV prefix sums. For round-robin KV ownership, local position `j` maps to global position `j * world_size + rank`; causal masking must use that global position.

## Partial attention and merge

Each rank's attention output is normalized over its own visible KV. The [round-robin reference](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_mla_persistent_round_robin.py) merges rank outputs using their log-sum-exp values. Summing or averaging already normalized outputs gives a different result. Empty shards use zero output and negative-infinity LSE in the reference.

The reference constructs rank-local cases and merges their results; its presence is not evidence of an end-to-end multi-GPU serving measurement. Distinguish local attention, transport and final merge when reporting latency. Also check each selected MLA branch: exposing CP arguments at the top-level API does not mean every dtype/layout branch supports them.

## Indexer connection

[Sparse block selection](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/sparse_attn/msa/pa_sparse_block_select.hpp) defines a context-parallel selector that nominates a rank's shard, publishes candidates through IPC and merges peers into a sparse page table. This is a separate selection stage ahead of sparse attention, not the MLA output merge above. Its [multi-GPU test](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/multigpu_tests/test_msa_block_select_cp.py) is the appropriate source entry for the IPC path.

## Evidence and validation

- [op_tests/test_mla_persistent_round_robin.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_mla_persistent_round_robin.py)
- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/multigpu_tests/test_msa_block_select_cp.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
