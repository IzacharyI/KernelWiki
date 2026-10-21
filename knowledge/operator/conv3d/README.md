# conv3d

2 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| flydsl | [Three-dimensional convolution — AITER flydsl](backend/flydsl_aiter.md) |
| triton | [Three-dimensional convolution — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

AITER contains implicit-GEMM, im2col, transpose and specialized 3D convolution paths. The cost of input preparation can dominate a fast inner GEMM.

## Validation

Record tensor layout, kernel size, stride, padding and dilation. Include transforms/im2col in the full operator timing.
