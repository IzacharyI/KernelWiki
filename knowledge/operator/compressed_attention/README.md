# compressed_attention

1 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| flydsl | [Compressed attention — AITER flydsl](backend/flydsl_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

Compression changes the representation consumed by attention; fused paths combine preparation and attention steps. Compression ratio, positional features and cache layout must match the source wrapper.

## Validation

Validate the compression result and final attention output. Include preparation cost when comparing against a decomposed reference.
