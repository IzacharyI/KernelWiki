# groupnorm

2 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Group normalization — AITER aiter](backend/aiter_aiter.md) |
| hip | [Group normalization — AITER hip](backend/hip_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

GroupNorm reduces within channel groups and spatial dimensions. Layout and group size control memory locality and reduction width.

## Validation

Match channel order, number of groups, epsilon, affine parameters and every spatial dimension.
