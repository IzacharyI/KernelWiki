---
title: "Paged kv copy: AITER triton"
kind: implementation_knowledge
operator: paged_kv_copy
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Paged kv copy — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py) · **Recent**

Source documentation: Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension the concatenated k_nope and k_pe are copied to kv_cache inplace Key parameters: - q_nope: Matrix X with shape (B, QH, D1). - q_pe: Matrix W with shape (B, QH, D2). - k_nope: Matrix X with shape (B_slot, KH, D1). - k_pe: Matrix W with shape (B_slot, KH, D2). - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2). - slot_mapping: Matrix W with shape (B_slot, ).

Symbols: [`fused_qk_rope_cat_and_cache_mla_fake_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L46), [`fused_qk_rope_cat_and_cache_mla`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L106), [`fused_qk_cat_and_cache_mla_fake_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L197), [`fused_qk_cat_and_cache_mla`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L234), [`_cat_and_cache_mla`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L302), [`fused_qk_rope_reshape_and_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L573), [`fused_qk_rope_cosine_cache_llama`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_kv_cache.py#L885).

Parameters referenced in this file: `BLOCK_H`, `BLOCK_H_HT`, `BLOCK_SIZE`, `BLOCK_T`, `num_warps`.

### `aiter/ops/triton/kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kv_cache.py) · **Recent**

Source documentation: Perform concat k_nope and k_pe to kv_cache inplace Key parameters: - k_nope: Matrix X with shape (B_slot, KH, D1). - k_pe: Matrix W with shape (B_slot, KH, D2). - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2). - slot_mapping: Matrix W with shape (B_slot, ).

Symbols: [`cat_and_cache_mla_fake_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kv_cache.py#L15), [`cat_and_cache_mla`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kv_cache.py#L28).

Parameters referenced in this file: `BLOCK_SIZE`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py) · **Recent**

Symbols: [`_store_kv_cache_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L16), [`_unit_cat`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L156), [`_unit_rope_cat`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L191), [`_fused_qk_rope_cat_and_cache_mla_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L271), [`_unit_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L581), [`_unit_rope_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L609), [`_fused_qk_rope_reshape_and_cache_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L662), [`_fused_qk_rope_cosine_cache_llama_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L1007).

Parameters referenced in this file: `BLOCK_D1`, `BLOCK_H`, `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kv_cache.py) · **Recent**

Symbols: [`_store_mla_kv_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kv_cache.py#L8), [`_cat_and_cache_mla_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kv_cache.py#L148).

Parameters referenced in this file: `BLOCK_SIZE`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_cache_copy.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_cache_copy.py)
- [op_tests/triton_tests/fusions/test_fused_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_kv_cache.py)
- [op_tests/triton_tests/test_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_kv_cache.py)
- [op_tests/test_kvcache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_kvcache.py)
- [op_tests/test_kvcache_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_kvcache_blockscale.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
