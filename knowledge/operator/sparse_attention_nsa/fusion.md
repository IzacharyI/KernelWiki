---
title: sparse_attention_nsa — fusion
kind: technique
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
---

# sparse_attention_nsa — fusion

## What fuses (and what deliberately doesn't)
| fusion | where | note |
|---|---|---|
| **selected + sliding-window branches** | one Triton kernel | FLA ships a fused kernel combining selected attention with the sliding branch (2025-02 release) |
| **compressed-KV pooling** | pre-step | block-mean/learned pooling into block summaries; feeds both the compressed branch and the selection score |
| **indexer / fp8 MQA logits** | the selection score | DeepSeek runs it in fp8; its output ranks blocks (top-k). Kept as its own kernel (it's the hot path) |
| **online top-k** | inside the indexer epilogue | avoids materializing the full score matrix (FLA) |
| **fnuz quantise-and-insert KV** | KV-cache write | fused quantise + paged insert so cache bytes match the fnuz dialect (V4 bring-up) |
| **RoPE / qk-norm** | before scoring | standard ([rope](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rope/README.md), [rmsnorm](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rmsnorm/README.md)) |
| **gather of selected blocks** | the selected-branch load | indexed/masked `buffer_load`; the non-contiguous gather is the perf-sensitive seam ([gather_scatter](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/gather_scatter/README.md)) |

## What FSA deliberately *splits* (anti-fusion)
Flash Sparse Attention found that **decoupling** is faster than one mega-kernel for the selected branch:
- online-softmax statistics → a **separate pre-compute kernel**;
- attention accumulation → **another kernel**;
- plus an **early-return** on empty index tensors.
This avoids loop-order stalls from non-contiguous query access — so for the selected branch, *less*
fusion can win on AMD. Benchmark both.

## Cross-links
- Sliding branch: [sliding_window_attention](../sliding_window_attention/README.md) · selection: [argmax_topk](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/argmax_topk/README.md) · gather: [gather_scatter](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/gather_scatter/README.md) /
  [paged_kv_copy](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/paged_kv_copy/README.md) · KV quant: [kv_cache_quant](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/kv_cache_quant/README.md) · MLA base: [mla_attention](../mla_attention/README.md).
- Languages: `triton_amd` (portable path), `hip_cpp` (gather/indexer helpers), `tilelang`.
- aiter dispatcher: `aiter`.

## Sources
- FLA fused selected+sliding, online top-k: https://github.com/fla-org/native-sparse-attention
- FSA decoupled softmax/accumulate + early-return: https://arxiv.org/html/2508.18224v1
- fnuz quantise-and-insert KV: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/sparse_attention_nsa/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
