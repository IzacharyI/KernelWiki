---
title: "Paged attention: cache contracts and preparation fusion"
kind: implementation_knowledge
operator: attention_decode_paged
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# KV-cache preparation belongs to the attention workload

Cache layout and quantization decide which attention implementation can consume a sequence. Keep these contracts with paged attention instead of comparing cache-copy or quantization kernels in isolation.

The [AITER cache API](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/cache.py) provides `reshape_and_cache`, a flash-layout variant, per-token quantized writes and block-quantized writes. The arguments expose slot mapping, separate K/V scales and, on selected paths, an ASM layout flag. These are distinct producer/consumer contracts rather than interchangeable storage operations.

The [Triton paged-decode wrapper](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode.py) selects attention variants with quantization support. Follow the selected wrapper into the device kernel before choosing KV layout, split count or scale granularity.

For fused preparation, consult [Q/K normalization, RoPE and cache writes](../../qk_norm_rope_cache/README.md) and the [fused BMM/RoPE/cache implementation](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py). A fusion must preserve Q/K head mapping, position convention, scale interpretation and every modified cache location.

Benchmark both the attention body and the full cache-write-to-attention chain. Cover nonuniform sequence lengths, partial pages, padded slots, grouped query heads and quantized-cache accuracy over successive decode steps. Record the page size and physical strides with results, so an inner-kernel win can be reproduced in the serving workload.

These are pinned source references. No new GPU timings or correctness results were collected for this page.
