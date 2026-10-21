# cross_entropy

1 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| triton | [Cross entropy — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

Vocabulary-wide reductions combine log-sum-exp with label selection, and may produce gradients. Ignore indices, reduction mode and numerical stability are observable semantics.

## Validation

Test extreme logits, ignored labels and backward results with the same reduction mode. Report vocabulary size and row count.
