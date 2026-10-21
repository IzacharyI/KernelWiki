---
title: layernorm — fusion neighbors
kind: technique
operator: layernorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/norm.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py
---

# layernorm — fusion

Identical fusion economics to [rmsnorm](../rmsnorm/README.md) (a bandwidth-bound anchor) — fold neighbors to share the single
read+write. aiter ships the full matrix:

## 1. residual-add + layernorm → (LayerNorm form of [fused_add_rmsnorm](../fused_add_rmsnorm/README.md))
`layernorm2d_fwd_with_add` / `_fused_add_layernorm` (Triton `layernorm2d_fwd_with_add`): `residual_out =
x + residual_in`, then normalize, one pass. The encoder/ViT analog of the decoder's fused-add-rmsnorm.

## 2. layernorm + fp8/int8 dynamic/smooth quant → [fused_norm_quant](../fused_norm_quant/README.md)
- `layernorm2d_fwd_with_dynamicquant` — per-token fp8 dynamic quant in the norm kernel.
- `layernorm2d_fwd_with_smoothquant` — SmoothQuant (per-channel scale) for int8.
- `layernorm2d_fwd_with_add_dynamicquant` / `_with_add_smoothquant` — residual+norm+quant triple-fusion.
Output is fp8/int8 + scale, halving downstream-GEMM input traffic. Cross-link [quant_dequant_fp8](../quant_dequant_fp8/README.md),
[quant_int8](../quant_int8/README.md), [fused_norm_quant](../fused_norm_quant/README.md).

## 3. groupnorm sibling
aiter also ships `groupnorm.py` (vision/diffusion); same reduction machinery over channel groups — out of
scope here but shares the wave64 reduce primitive.

## Fusion table
| form | aiter entrypoint | saves |
|---|---|---|
| add + LN | `layernorm2d_fwd_with_add` | residual round-trip |
| LN + dyn-quant | `layernorm2d_fwd_with_dynamicquant` | ½ output bytes (fp8) |
| LN + smoothquant | `layernorm2d_fwd_with_smoothquant` | int8 GEMM input |
| add + LN + quant | `layernorm2d_fwd_with_add_dynamicquant` | both |

## torch.compile
Same as rmsnorm: register fused ops as custom ops so Inductor fuses around them; ROCm fusion pass stitches
norm+quant. See [backends/vllm_kernels/aiter_integration](../../general/libraries/vllm_kernels/aiter_integration.md).

## Sources
- aiter fused layernorm variants: `/sgl-workspace/aiter/aiter/ops/norm.py` (`layernorm2d_fwd_with_add`, `_with_smoothquant`, `_with_add_smoothquant`).
- Triton fused layernorm bodies: `/sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py` (`layernorm2d_fwd_with_add`, `_with_dynamicquant`, `_with_add_dynamicquant`).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/layernorm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
