---
title: alibi — fusion neighbors
kind: technique
operator: alibi
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2108.12409
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
---

# alibi — fusion

ALiBi is **defined by being fused into attention** — it has no standalone existence on the serving path.

## 1. Folded into the attention score (the op)
The ALiBi bias is added to `S_{ij}` inside the FMHA inner loop, before softmax — it never materializes.
This is true for prefill ([attention_prefill_fmha](../attention_prefill_fmha/README.md)), decode/paged ([attention_decode_paged](../attention_decode_paged/README.md)), and
GQA/MQA ([gqa_mqa_attention](../gqa_mqa_attention/README.md)). The "fusion" is the only form.

## 2. Mutually exclusive with RoPE
A model uses **either** ALiBi **or** [rope](../rope/README.md)/[mrope](../mrope/README.md) for positional information, not both. So ALiBi and
the RoPE-family fusions never coexist in the same attention entry — picking the model's scheme is a
config-level choice.

## 3. Backend support, not stacking
Unlike the norm/quant fusions, ALiBi doesn't *stack* with more ops — it's a single bias term. The practical
question is whether the chosen attention backend exposes an ALiBi bias path:
- vLLM custom paged-attn HIP (`csrc/rocm/attention.cu`) supports an ALiBi slopes arg.
- Triton FMHA kernels take a bias / alibi-slopes argument.
- CK/aiter FMHA: check the kernel signature for an alibi/bias parameter for your shape.

## Fusion table
| form | where | note |
|---|---|---|
| ALiBi bias in score | inside FMHA | the only form |
| (vs RoPE) | attention entry | mutually exclusive |

## Sources
- ALiBi inside attention: https://arxiv.org/abs/2108.12409.
- vLLM paged-attn ALiBi slopes: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/alibi/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
