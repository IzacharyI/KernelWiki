---
title: sliding_window_attention — fusion
kind: technique
operator: sliding_window_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# sliding_window_attention — fusion

## What fuses into the SWA kernel
SWA is FMHA + band mask, so it inherits the full-attention fusion neighbors and adds a few SWA-specific
ones:

| fusion | where | benefit |
|---|---|---|
| **RoPE / qk-norm pre-step** | before QKᵀ | one less HBM round-trip; standard FA fusion ([rope](../rope/README.md), [rmsnorm](../rmsnorm/README.md)) |
| **Band mask + block-skip** | the KV loop bound itself | the SWA win — see [tuning.md](tuning.md) |
| **Logit soft-cap** | fp32, before mask add | Gemma-2/3 (`tanh` cap); fused into the score |
| **Attention sink** | mask coordinate builder | StreamingLLM / GPT-OSS keep first-N + last-W |
| **fp8 KV-cache quant/dequant** | KV load | bandwidth; fnuz on gfx942 ([kv_cache_quant](../kv_cache_quant/README.md)) |
| **Output projection** | epilogue (rarely) | usually a separate GEMM; kept distinct on AMD |

## Hybrid-model layer routing (the macro-fusion)
The bigger "fusion" is architectural: SWA-heavy hybrids (Gemma-2 1:1 SWA/full, Qwen "every N", Mistral)
share the same KV cache manager but route per-layer to SWA vs full attention. On ROCm, getting both
paths on the **same backend** (so the scheduler/graph capture is uniform) matters more than micro-fusion.
vLLM's `ROCM_AITER_FA` routes Decode/Prefill/Extend per request; SWA layers ride the same dispatcher.

## Cross-links
- Core attention fusion: [attention_prefill_fmha](../attention_prefill_fmha/README.md) · [attention_decode_paged](../attention_decode_paged/README.md)
- Chunked/prefix-cache interplay: [chunked_prefill](../chunked_prefill/README.md)
- KV quant: [kv_cache_quant](../kv_cache_quant/README.md) · RoPE: [rope](../rope/README.md)

## Sources
- FA SWA + soft-cap + sink: https://github.com/Dao-AILab/flash-attention
- MI300X 2-GEMM attention fusion / OPTIMIZE_EPILOGUE: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/sliding_window_attention/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
