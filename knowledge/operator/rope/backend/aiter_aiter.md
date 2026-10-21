---
title: "Rope: AITER aiter"
kind: implementation_knowledge
operator: rope
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Rope — AITER aiter

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/inverse_rope_group_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/inverse_rope_group_quant.py) · **Recent**

Source documentation: DeepSeek-V4 attention-output inverse RoPE fused with FP8 group quant. Its own JIT module: this sits on the attention *output* path and shares no kernel with the QK-norm/RoPE input-path ops, so folding it into their 6k-line translation unit would only make every edit there rebuild it.

Symbols: [`_check_layout_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/inverse_rope_group_quant.py#L29), [`_inverse_rope_group_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/inverse_rope_group_quant.py#L54), [`scale_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/inverse_rope_group_quant.py#L67), [`inverse_rope_group_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/inverse_rope_group_quant.py#L91).

### `aiter/ops/pos_encoding.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pos_encoding.py) · **Recent**

Symbols: [`rotary_embedding_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pos_encoding.py#L12), [`batched_rotary_embedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/pos_encoding.py#L25).

### `aiter/ops/rope.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py) · **Recent**

Source documentation: Forward propagation of traditional RoPE (Rotary Position Embedding). Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d // 2] if reuse_freqs_front_part is true. Otherwise, it should be in [s, 1, 1, d]. rotate_style: 0 - NEOX style which rotates the 2nd half of elements, 1 - GPT-J style which rotates odd part. When rotate dim is smaller than d, front part is just copied if nope_first is true, or later part is copied if nope_first is false.

Symbols: [`rope_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L12), [`rope_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L31), [`rope_2c_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L50), [`rope_2c_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L71), [`rope_cached_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L92), [`rope_cached_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L112), [`rope_cached_2c_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L132), [`rope_cached_2c_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L154), [`rope_cached_positions_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L176), [`rope_cached_positions_2c_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L198), [`rope_cached_positions_offsets_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L222), [`rope_cached_positions_offsets_2c_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L245), [`rope_thd_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L270), [`rope_thd_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L291), [`rope_2d_fwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L312), [`rope_2d_bwd_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L339), [`rope_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L365), [`rope_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L387), [`rope_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L397), [`rope_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L427), [`rope_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L471), [`rope_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L491), [`rope_cached_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L535), [`rope_cached_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L558), [`rope_cached_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L571), [`rope_cached_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L603), [`rope_cached_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L649), [`rope_cached_2c_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L671), [`rope_cached_positions_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L717), [`rope_cached_positions_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L748), [`rope_cached_positions_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L796), [`rope_cached_positions_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L817), [`rope_cached_positions_offsets_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L841), [`rope_cached_positions_offsets_2c_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L874), [`rope_cached_positions_offsets_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L924), [`rope_cached_positions_offsets_2c_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L947), [`rope_thd_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L973), [`rope_thd_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L994), [`rope_thd_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1013), [`rope_2d_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1034), [`rope_2d_fwd_inplace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1063), [`rope_2d_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1090), [`RoPE`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1119), [`RoPECached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1156), [`RoPETHD`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1201), [`RoPE2D`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/rope.py#L1237).

### `aiter/rotary_embedding.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py) · **Recent**

Source documentation: Rotary Positional Embeddings.

Symbols: [`_get_rope_param`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L47), [`_rotate_neox`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L67), [`_rotate_gptj`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L73), [`_apply_rotary_emb`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L80), [`RotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L110), [`LinearScalingRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L461), [`DynamicNTKScalingRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L548), [`_yarn_find_correction_dim`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L590), [`_yarn_find_correction_range`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L602), [`_yarn_linear_ramp_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L618), [`_yarn_get_mscale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L629), [`YaRNScalingRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L635), [`Phi3LongRoPEScaledRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L704), [`yarn_get_mscale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L838), [`DeepseekScalingRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L844), [`Llama3RotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L935), [`apply_interleaved_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L982), [`MRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L993), [`AiterFusedSetKVBufferArg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1193), [`RotaryEmbeddingFusedQKNorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1204), [`MRotaryEmbeddingQKNormFused`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1361), [`DualChunkRotaryEmbedding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1524), [`get_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1723), [`get_rope_wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/rotary_embedding.py#L1961).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_indexer_qk_rope_quant_and_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_indexer_qk_rope_quant_and_cache.py)
- [op_tests/test_inverse_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_inverse_rope_group_quant.py)
- [op_tests/test_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_rope.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_rope_reshape_and_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_rope_reshape_and_cache.py)
- [op_tests/op_benchmarks/triton/bench_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_rope.py)
- [op_tests/triton_tests/rope/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/__init__.py)
- [op_tests/triton_tests/rope/test_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_rope.py)
- [op_tests/triton_tests/rope/test_rope_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_rope_mxfp4.py)
- [op_tests/triton_tests/torch_compile/test_compile_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_rope.py)
- [op_tests/triton_tests/utils/rotary_embedding.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/rotary_embedding.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
