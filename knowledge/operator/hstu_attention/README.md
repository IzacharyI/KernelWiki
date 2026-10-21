# hstu_attention

2 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| flydsl | [HSTU attention — AITER flydsl](backend/flydsl_aiter.md) |
| triton | [HSTU attention — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

HSTU attention has its own score transform and masking semantics, with separate forward and backward implementations. A standard softmax-attention reference is not interchangeable.

## Validation

Use AITER HSTU tests for the selected mask and jagged sequence layout. Validate gradients separately from forward output.
