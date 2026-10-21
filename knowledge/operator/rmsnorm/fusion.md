---
title: rmsnorm — fusion neighbors
kind: technique
operator: rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py
  - https://github.com/vllm-project/vllm/pull/14959
  - https://github.com/sgl-project/sglang/issues/18466
---

# rmsnorm — fusion

RMSNorm is a **fusion anchor**, not a standalone op on the serving path. Because it is bandwidth-bound,
every neighbor you fold in removes a full HBM round-trip of the residual stream. The fusion ladder, from
most to least common in production:

## 1. residual-add + rmsnorm → [fused_add_rmsnorm](../fused_add_rmsnorm/README.md)  (the dominant form)
Transformer blocks do `h = h + sublayer(norm(h))`; the *next* norm reads `h`. Fusing the add into the
norm means `residual_out = x + residual_in` and `y = rmsnorm(residual_out)` in **one read + one write**
instead of add(write)→norm(read). aiter: `rmsnorm2d_fwd_with_add` / `add_rmsnorm`; vLLM:
`fused_add_rms_norm_kernel`; Triton: `_fused_add_rmsnorm_kernel`. This is what `VLLM_ROCM_USE_AITER_
RMSNORM=1` engages (vLLM PR #14959). See the dedicated card [fused_add_rmsnorm](../fused_add_rmsnorm/README.md).

## 2. rmsnorm + fp8/int8 dynamic quant → [fused_norm_quant](../fused_norm_quant/README.md)
The norm output feeds a quantized GEMM, so quantize *in the norm kernel* while `y` is in fp32 registers —
no extra read. aiter: `rmsnorm2d_fwd_with_dynamicquant`, `rmsnorm2d_fwd_with_smoothquant`,
`rmsnorm2d_fwd_with_add_dynamicquant` (residual+norm+quant triple-fusion), `add_rmsnorm_quant`,
`gated_rmsnorm_fp8_group_quant`. SGLang reports **1–6% e2e latency** from RMSNorm+FP8-dynamic-quant on
Qwen3 MI300X (#18466). Cross-link [quant_dequant_fp8](../quant_dequant_fp8/README.md) / [quant_int8](../quant_int8/README.md). See [fused_norm_quant](../fused_norm_quant/README.md).

## 3. QK-norm + RoPE + KV-write + quant → the attention-entry mega-fusion
Qwen3 / Qwen3-VL apply RMSNorm to Q and K *before* RoPE. aiter fuses the whole attention entry:
`fused_qk_norm_rope_cache_quant` (and the mrope variant `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`)
does QK-RMSNorm → [rope](../rope/README.md) → KV-cache write → optional fp8 quant in one kernel. SGLang's Qwen3 work cites
this fused QKNorm+RoPE+KV-set kernel as a significant per-layer win (#18466). Cross-link [rope](../rope/README.md),
[mrope](../mrope/README.md), [kv_cache_quant](../kv_cache_quant/README.md).

## 4. all-reduce + rmsnorm → [fused_allreduce_rmsnorm](../fused_allreduce_rmsnorm/README.md)  (TP)
In tensor-parallel, the post-attention/MLP all-reduce is immediately followed by the next RMSNorm.
Fusing them (aiter custom-AR + norm, or the one-shot AR+norm path) hides the norm under the collective.
Cross-link [fused_allreduce_rmsnorm](../fused_allreduce_rmsnorm/README.md), [allreduce](../allreduce/README.md).

## Fusion economics (why it works)
| form | HBM passes (x) | wins |
|---|---|---|
| add → norm (unfused) | write h, read h, read x, write y = 4 | baseline |
| fused add+norm | read x, read resid, write resid, write y = 4 but no h round-trip | ~1 pass saved |
| +dynamic quant | quant in-register, write fp8 (½ bytes) | ½ output traffic |
| +AR (TP) | norm hidden under collective latency | ~free |

## torch.compile interaction
On vLLM, the fused norm ops are registered as custom ops (`direct_register_custom_op`) so Inductor fuses
*around* them instead of decomposing — and a ROCm fusion pass
(`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`) stitches rms+quant chains in the compiled graph.
Don't register → Inductor regenerates generic Triton and you lose the aiter kernel. See
[backends/vllm_kernels/aiter_integration](../../general/libraries/vllm_kernels/aiter_integration.md).

## Sources
- aiter fused variants (`rmsnorm2d_fwd_with_add`, `_with_dynamicquant`, `add_rmsnorm_quant`, `_ck`): `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`.
- QK-norm+RoPE+KV+quant fusion: `/sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py`, `fused_qk_norm_mrope_cache_quant.py`.
- vLLM AITER RMSNorm (with_add) integration: https://github.com/vllm-project/vllm/pull/14959.
- 1–6% e2e from RMSNorm+FP8 quant fusion on Qwen3: https://github.com/sgl-project/sglang/issues/18466.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/rmsnorm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
