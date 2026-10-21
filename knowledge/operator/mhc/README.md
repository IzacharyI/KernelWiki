# mhc

5 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Manifold-constrained hyper-connections — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Manifold-constrained hyper-connections — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Manifold-constrained hyper-connections — AITER gluon](backend/gluon_aiter.md) |
| hip | [Manifold-constrained hyper-connections — AITER hip](backend/hip_aiter.md) |
| triton | [Manifold-constrained hyper-connections — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

MHC paths implement pre/post residual mixing and normalization-related stages, including Sinkhorn-style transforms in selected entries. Fusion changes intermediate storage and reduction order.

## Validation

Match channel count, mixing matrices, iteration settings and all residual outputs. Include surrounding pre/post work in a fused comparison.
