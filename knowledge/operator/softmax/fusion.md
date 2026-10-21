---
title: softmax — fusion neighbors
kind: technique
operator: softmax
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/topk.py
  - /sgl-workspace/aiter/aiter/ops/triton/softmax.py
  - https://github.com/vllm-project/vllm/pull/16752
---

# softmax — fusion

Softmax is **defined by its fusions** — standalone it's negligible. The three that matter:

## 1. Into attention (online / flash softmax) — the whole point
The QKᵀ→softmax→PV inner loop never materializes the full softmax; it runs an **online softmax** with a
running `(m, l)` so the score matrix stays in registers/LDS. This is the dominant softmax on any LLM and
lives entirely inside FMHA. Optimize the attention kernel, not softmax. See [attention_prefill_fmha](../attention_prefill_fmha/README.md),
[attention_decode_paged](../attention_decode_paged/README.md), [mla_attention](../mla_attention/README.md).

## 2. softmax + topk → MoE routing
MoE routers compute `topk(softmax(logits))` (or softmax over the top-k). aiter fuses this:
`topk_softmax` (and vLLM's `rocm_aiter_topk_softmax`, PR #16752; biased/grouped variants for DeepSeek).
One kernel: softmax over n_experts + top-k selection + renormalize. Cross-link [moe_routing_topk](../moe_routing_topk/README.md),
[argmax_topk](../argmax_topk/README.md).

## 3. softmax + sampling → logits → token
After the lm_head, `softmax(logits)` feeds top-p/top-k sampling. Often kept separate (vocab is wide) but
the softmax+sort can fuse for greedy/argmax. Cross-link [sampling_topk_topp](../sampling_topk_topp/README.md), [lm_head_logits](../lm_head_logits/README.md).

## Fusion table
| form | where | op |
|---|---|---|
| online softmax | inside FMHA | [attention_prefill_fmha](../attention_prefill_fmha/README.md) |
| softmax + topk | MoE router | aiter `topk_softmax` → [moe_routing_topk](../moe_routing_topk/README.md) |
| softmax + sample | post-lm_head | → [sampling_topk_topp](../sampling_topk_topp/README.md) |

## Sources
- aiter topk_softmax fusion: `/sgl-workspace/aiter/aiter/ops/topk.py`; vLLM registered op: https://github.com/vllm-project/vllm/pull/16752.
- online softmax in attention: perf_knowledge [attention_prefill_fmha](../attention_prefill_fmha/README.md).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/softmax/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
