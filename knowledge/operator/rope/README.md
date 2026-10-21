# rope

3 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Rope — AITER aiter](backend/aiter_aiter.md) |
| hip | [Rope — AITER hip](backend/hip_aiter.md) |
| triton | [Rope — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the AITER pages above for the current source snapshot.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: rope — overview
kind: operator_overview
operator: rope
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rope.py
  - /sgl-workspace/aiter/aiter/ops/triton/rope/rope.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
```

</details>

# rope  (rotary position embedding: rotate Q/K by position-dependent angles)

## TL;DR
RoPE applies a position-dependent **2D rotation** to pairs of head dimensions of Q and K, just before
attention. It is **memory-bound** (read Q/K, read cos/sin cache, write Q/K) and runs **once per layer**.
The dominant optimization is **fusion into the attention entry** (QK-norm + RoPE + KV-cache write + quant
in one kernel) — done standalone it was 40–60% of inference latency in a naive PyTorch impl (vLLM #16457
moved it to Triton for that reason). Cross-link [mrope](../mrope/README.md) (3D/multimodal) and [fused_norm_quant](../fused_norm_quant/README.md).

## Math contract
For position `p` and dim pair `(2i, 2i+1)` with angle `θ_i = p · base^(−2i/d)`:
`x'_{2i} = x_{2i}cosθ_i − x_{2i+1}sinθ_i`, `x'_{2i+1} = x_{2i}sinθ_i + x_{2i+1}cosθ_i`. Two styles:
**NeoX** (rotate-halves: pair `i` with `i+d/2`) and **GPT-J** (rotate adjacent pairs) — `is_neox` flag.
Inputs: `q[*, h, d]`, `k[*, h, d]`, `cos_sin_cache[max_pos, d]` (precomputed). dtype bf16/fp16, cos/sin
fp32. Partial rotation: only the first `rotary_dim ≤ head_size` dims are rotated (GLM/partial models).

## Shape regimes
- **prefill**: `[batch, seq, h, d]`, seq 1k–64k → many elements, fills the chip.
- **decode**: `[batch, 1, h, d]` (one token) → few elements, latency-bound; fuse to hide it.
- `d` = head_size (64/128); `rotary_dim` may be < d (partial). `h` = n_heads (Q) / n_kv_heads (K).

## Where it matters (Amdahl)
Standalone RoPE is 1–3% GPU time when kernelized, but a **naive PyTorch RoPE was 40–60% of latency** for
Qwen2-VL (vLLM #16457) — the lesson is that RoPE *must* be a fused kernel. Done right + fused into the
attention entry it's near-free.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (CK/asm rope_fwd family + fused QK-norm+RoPE) | [backends/aiter.md](backend/aiter.md) |
| triton | 🟢 sota (aiter Triton rope + vLLM Triton rope) | [backends/triton.md](backend/triton.md) |
| hip | 🟢 sota (vLLM `rotary_embedding` HIP) | [backends/hip.md](backend/hip.md) |
| vllm_kernels | 🟢 sota (HIP pos_encoding + AITER rope) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |

## Fusion neighbors
`QK-norm + RoPE + KV-cache write + quant` → aiter `fused_qk_norm_rope_cache_quant` ([fused_norm_quant](../fused_norm_quant/README.md),
[kv_cache_quant](../kv_cache_quant/README.md)); `+KV cache write` (`rope_cached_*`, fused_kv_cache); inside attention prefill.
3D/multimodal variant → [mrope](../mrope/README.md). See [fusion.md](fusion.md).

## Numerics
cos/sin in fp32; NeoX vs GPT-J style must match the checkpoint; partial-rotation bounds; scaling variants
(YaRN/NTK/linear) change the angle table. See [numerics.md](numerics.md).

## How to bench
`op_tests/test_rope.py` (aiter) at `[b, s, h, d]`; oracle = fp64 reference rotation; e2e A/B with the fused
QK-norm+RoPE path.

## Sources
- aiter rope ops (cached/uncached, 1c/2c, thd, 2d, positions/offsets): `/sgl-workspace/aiter/aiter/ops/rope.py`.
- aiter Triton rope (`_rope_fwd`, BLOCK_D, BLOCK_S=32, num_warps=4): `/sgl-workspace/aiter/aiter/ops/triton/rope/rope.py`.
- vLLM HIP `rotary_embedding` (is_neox, cos_sin_cache): https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- naive RoPE 40–60% latency → Triton: https://github.com/vllm-project/vllm/pull/16457.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/rope/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
