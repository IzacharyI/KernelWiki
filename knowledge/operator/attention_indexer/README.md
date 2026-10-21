# attention_indexer

6 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Attention index scoring and selection — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Attention index scoring and selection — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Attention index scoring and selection — AITER gluon](backend/gluon_aiter.md) |
| hip | [Attention index scoring and selection — AITER hip](backend/hip_aiter.md) |
| opus | [Attention index scoring and selection — AITER opus](backend/opus_aiter.md) |
| triton | [Attention index scoring and selection — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

## Validation

Check selected indices as well as logits, including ties and masked pages. Keep scoring and top-k costs visible when evaluating sparse attention.
