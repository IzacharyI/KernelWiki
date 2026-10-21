---
title: "Prefill and full attention: AITER triton"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER triton

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/chunked_pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunked_pa_prefill.py) · **Recent**

Source documentation: Unified attention for mixed prefill (multi-token) and decode (single-token) sequences with paged KV cache. Args: query (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim). key (torch.Tensor): Key tensor for prefill portion with shape (total_tokens, num_kv_heads, head_dim). value (torch.Tensor): Value tensor for prefill portion with shape (total_tokens, num_kv_heads, head_dim).

Symbols: [`chunked_prefill_paged_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunked_pa_prefill.py#L29).

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/dsv4_indexer.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/dsv4_indexer.py) · **Recent**

Source documentation: Compute indexer scores. Args: q: [S, H, Hd] bf16 k: [P, Hd] bf16 w: [S, H] fp32 compress_ratio: int Returns: scores: [S, P] fp32

Symbols: [`_causal_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/dsv4_indexer.py#L11), [`indexer_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/dsv4_indexer.py#L17), [`dsv4_indexer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/dsv4_indexer.py#L71).

### `aiter/ops/triton/attention/extend_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/extend_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supports page size = 1 and prefill with KV cache (i.e. extend).

Symbols: [`extend_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/extend_attention.py#L34), [`redundant_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/extend_attention.py#L164).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DPE`, `BLOCK_DV`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/attention/fav3_sage.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py) · **Recent**

Source documentation: Return Triton launch config for fav3_sage sage_fwd. Default behaviour is unchanged from upstream aiter (gfx942: num_warps=8, etc.). Set a3_tuned=True to opt into MI308/gfx942 perf tuning for 128/64 tiles (num_warps=4, waves_per_eu=2). Numerics match the default launch params; use this when long sequences (e.g. Qwen edit ~8400 tokens) are register-bound.

Symbols: [`_load_sage_fwd_tables`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py#L29), [`get_sage_fwd_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py#L70), [`_FAv3SageWrapperFunc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py#L100), [`fav3_sage_wrapper_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py#L280), [`fav3_sage_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py#L389).

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py) · **Recent**

Source documentation: Returns tuned config for MXFP4 on supported architectures.

Symbols: [`get_sage_fwd_configs_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py#L28), [`_FAv3SageMXFP4WrapperFunc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py#L38), [`fav3_sage_mxfp4_wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py#L181), [`fav3_sage_mxfp4_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py#L230).

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_R`.

### `aiter/ops/triton/attention/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py) · **Recent**

Source documentation: True when the Gluon MHA forward kernel can actually run on this device.

Symbols: [`is_gluon_available`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L46), [`_resolve_backend`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L57), [`mha_set_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L82), [`mha_set_use_int64_strides`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L91), [`_fwd_offsets_fit_int32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L97), [`mha_set_swizzle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L175), [`_get_sliding_window_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L190), [`gluon_forward_unsupported_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L194), [`check_gluon_forward_support`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L218), [`_get_softmax_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L224), [`_pack_attn_returns`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L228), [`_gluon_flash_attn_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L237), [`_flash_attn_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L495), [`_FlashAttnFunc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L796), [`flash_attn_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L981), [`_FlashAttnVarlenFunc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L1119), [`flash_attn_varlen_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L1325), [`flash_attn_with_kvcache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py#L1485).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_OUT`, `BLOCK_DMODEL_PE`, `BLOCK_DMODEL_PE_POW2`, `BLOCK_DMODEL_POW2`, `BLOCK_M`, `BLOCK_N`, `num_stages`.

### `aiter/ops/triton/attention/mha_v3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py) · **Recent**

Source documentation: FlashAttention v3 entry point.

Symbols: [`_FlashAttnV3Func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L14), [`flash_attn_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L159), [`_FlashAttnVarlenV3Func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L198), [`flash_attn_varlen_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L341), [`flash_attn_with_kvcache_fake_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L382), [`_flash_attn_with_kvcache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L456), [`flash_attn_with_kvcache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L527), [`_quantize_bshd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L649), [`_quantize_thd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L731), [`_FlashAttnFP8Wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L847), [`flash_attn_fp8_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L1034), [`_FlashAttnVarlenFP8Wrapper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L1127), [`flash_attn_varlen_fp8_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py#L1334).

### `aiter/ops/triton/attention/na3d_flash.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/na3d_flash.py) · **Recent**

Source documentation: 3D neighborhood flash attention launcher. Each query token at grid position (t, h, w) attends to the KT x KH x KW neighborhood centered on it, with inward border-shift so every window has exactly KT x KH x KW keys. Usage:: from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn out = na3d_flash_attn(q, k, v, kernel_size=(11, 11, 11)) # q, k, v: (B, T, H, W, NH, HD) bfloat16, Q already scaled by head_dim**-0.5 # out : (B, T, H, W, NH, HD) bfloat16

Symbols: [`na3d_flash_attn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/na3d_flash.py#L33).

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_Q`, `num_stages`, `num_warps`.

### `aiter/ops/triton/attention/pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill.py) · **Recent**

Source documentation: Paged attention prefill for multi-token context processing with paged KV cache. Supports variable-length sequences, GQA, FP8 quantization, ALiBi, and sliding window. Args: q (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim). k (torch.Tensor): Key tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim). v (torch.Tensor): Value tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim).

Symbols: [`context_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill.py#L26).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PADDED`, `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pod_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pod_attention.py) · **Recent**

Source documentation: POD (Prefill-On-Decode) fused attention for simultaneous prefill and decode execution. Launches persistent kernels that execute both operations concurrently on different CUs for improved hardware utilization. Args: cu_ctr (torch.Tensor): CU (Compute Unit) counter for workload distribution. q (torch.Tensor): Decode query with shape (batch_size * 1, num_heads, head_dim). k (torch.Tensor): Decode key with shape (total_tokens, num_heads, head_dim).

Symbols: [`pod_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pod_attention.py#L20), [`get_num_splits_and_buffer_sizes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pod_attention.py#L292).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/prefill_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/prefill_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supporst page size = 1.

Symbols: [`context_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/prefill_attention.py#L33).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`.

### `aiter/ops/triton/attention/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/utils.py) · **Recent**

Source documentation: Convert a dense block attention mask to a ragged look-up table of KV block indices per (batch, head, q_block). Used for block-sparse attention with no per-iteration branching in the kernel. block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks) for per-head masks. True = may attend, False = must not attend. num_heads: Required when block_attn_mask is 3D (number of Q heads).

Symbols: [`block_attn_mask_to_ragged_lut`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/utils.py#L13).

Parameters referenced in this file: `BLOCK_KB`.

### `aiter/ops/triton/fusions/attn_res.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py) · **Recent**

Source documentation: Attention-residual (residual candidate gating) forward.

Symbols: [`_tensor_spec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L96), [`_fresh_tensor_spec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L114), [`_int_spec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L129), [`_next_pow2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L139), [`_dtype_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L151), [`_pick_attn_res_separate_bl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L260), [`_get_attn_res_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L276), [`_pick_attn_res_separate_num_warps`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L282), [`_pick_attn_res_seq_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L292), [`_fast_reshape2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L302), [`_fast_reshape3d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L313), [`_fast_flatten1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L319), [`_pick_attn_res_packed_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L325), [`_launch_tune_kwargs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L333), [`_assert_launch_cache_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L407), [`_launch_hooks_idle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L419), [`_launch_attn_res`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L431), [`_build_ptr_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L488), [`attn_res_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L501), [`_run_sequence`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L545), [`_run_packed`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L612), [`attn_res_gate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py#L680).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/quant/mxfp6_fmha_pack.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py) · **Recent**

Source documentation: MXFP6-E2M3 packing utilities for the gfx950 MHA v4 kernels.

Symbols: [`fp6_k_raw_buffer_sizes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L36), [`_k_lds_order_gather_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L52), [`_v_field_perm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L115), [`_v_kvtab_dev`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L131), [`quantize_fp6_v_clean_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L142), [`quantize_fp6_v_data_scale_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L179), [`_v_noswap_kvtab`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L219), [`_v_direct_kvtab`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L241), [`_qk_field_perm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L379), [`_qk_field_perm_dev`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L535), [`quantize_fp6_lastdim_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L544), [`_k_lds_gather_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L580), [`_k_lds_src_within`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L604), [`reorder_fp6_k_lds_order_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L619), [`quantize_fp6_k_lds_order_triton`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L716), [`fp6_k_lds_order_views_from_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L732), [`_qk_field_perm_pt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L756), [`_e2m3_encode_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L767), [`quantize_fp6_lastdim_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L789), [`_k_scale_tail_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L822), [`quantize_fp6_k_lds_order_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L856), [`pack_fp6_v_data_scale_views`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py#L908).

Parameters referenced in this file: `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/quant/sage_attention_quant_wrappers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py) · **Recent**

Source documentation: Round a V sequence length up to the 128-token FP4 packing tile.

Symbols: [`_bshd_order`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L24), [`sage_quant_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L32), [`_apply_int8_q_smoothing`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L146), [`_f4f4_v_kperm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L220), [`fp4_v_padded_sequence`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L242), [`fp4_v_raw_buffer_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L247), [`sage_quant_v_f4f4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L255), [`pack_v_mxfp4_colmajor_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L312), [`_pack_v_mxfp4_colmajor_raw_fake`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L359), [`sage_quant_v_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L370), [`sage_quant_mxfp6`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L383), [`sage_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L490), [`rotation_smooth_qk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L689), [`smooth_rotate_downcast_qk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L828), [`create_hadamard_matrix`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py#L1008).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `BLOCK_SIZE_M`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py) · **Recent**

Symbols: [`cdiv_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py#L20), [`_kernel_paged_attention_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py#L39).

Parameters referenced in this file: `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/attention/extend_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/extend_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supports page size = 1 and prefill with KV cache (i.e. extend).

Symbols: [`_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/extend_attention.py#L54), [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/extend_attention.py#L321).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DPE`, `BLOCK_DV`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py) · **Recent**

Source documentation: Calculate the window boundaries for a query block.

Symbols: [`map_dims`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L10), [`_sage_fwd_no_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L15), [`_sage_fwd_blocksparse_nomask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L224), [`_sage_fwd_blocksparse_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L397), [`_sage_fwd_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L573), [`compute_window_bounds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L934), [`classify_window_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L970), [`handle_padded_last_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L1004), [`compute_padding_info`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L1042), [`compute_block_masking`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L1062), [`sage_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py#L1243).

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py) · **Recent**

Source documentation: Calculate padding information for the last K block.

Symbols: [`compute_padding_info`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L8), [`compute_block_masking`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L28), [`_sage_fwd_no_mask_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L140), [`_sage_fwd_mask_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L244), [`_sage_fwd_blocksparse_nomask_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L355), [`_sage_fwd_blocksparse_mask_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L459), [`sage_fwd_mxfp4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py#L588).

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py) · **Recent**

Symbols: [`_cdiv_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L22), [`_load_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L27), [`_compute_alibi_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L45), [`_attn_fwd_inner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L80), [`_attn_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L304), [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py#L941).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PE`, `BLOCK_DMODEL_POW2`, `BLOCK_M`, `BLOCK_N`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/attention/na3d_flash.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/na3d_flash.py) · **Recent**

Source documentation: Triton flash-style 3D neighborhood attention. Each query token attends to the (KT x KH x KW) neighborhood centered on its (t, h, w) grid position, with inward border-shift so every window has the full KT x KH x KW keys. Algorithm: - Grid dim[0] = T × H × ⌈W / BLOCK_Q⌉; each program covers one (t, h) row. - Q is loaded once into registers and held for all KT x KH inner iterations. - Each iteration loads one (t_kv, h_kv) row of K and V at the shared W window.

Symbols: [`_get_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/na3d_flash.py#L28), [`_na3d_flash_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/na3d_flash.py#L49).

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_Q`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_prefill.py) · **Recent**

Symbols: [`_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_prefill.py#L29), [`_fwd_kernel_alibi`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_prefill.py#L318).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PADDED`, `BLOCK_M`, `BLOCK_MODEL`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/pod_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pod_attention.py) · **Recent**

Symbols: [`read_realtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pod_attention.py#L17), [`get_cu_id`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pod_attention.py#L32), [`pod_persistent`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pod_attention.py#L96).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/prefill_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/prefill_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supporst page size = 1.

Symbols: [`_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/prefill_attention.py#L44).

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/common/mha_kernel_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/mha_kernel_utils.py) · **Recent**

Symbols: [`_compute_fp8_scaling_factors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/mha_kernel_utils.py#L9).

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py) · **Recent**

Source documentation: Triton kernel helper functions shared across flash attention modules. This module contains Triton JIT-compiled helper functions that are used within the main attention kernels (fwd_prefill, fwd_decode, bwd). These are kept separate from utils.py to allow stricter type checking on pure Python utilities.

Symbols: [`compute_alibi_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L24), [`compute_fp8_scaling_factors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L45), [`_cast_varlen_to_fp8_kernel_2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L55), [`_rotary_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L159), [`cast_to_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L274), [`_apply_rotary_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L351), [`_ApplyRotary`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L431), [`apply_rotary_emb`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L489), [`apply_rotary`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py#L541).

Parameters referenced in this file: `BLOCK_H`, `BLOCK_K`, `BLOCK_M`, `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py) · **Recent**

Source documentation: Allowed query/key pairs; negative window sizes mean unbounded.

Symbols: [`get_fwd_prefill_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L37), [`attention_mask`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L258), [`_attn_fwd_inner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L284), [`compute_window_bounds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L522), [`classify_window_blocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L562), [`handle_padded_last_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L596), [`compute_padding_info`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L634), [`compute_block_masking`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L654), [`attn_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L816), [`attention_forward_prefill_triton_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py#L1365).

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_M_OPTIONS`, `BLOCK_N`, `BLOCK_N_OPTIONS`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py) · **Recent**

Symbols: [`fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py#L29), [`varlen_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py#L354), [`fwd_kvcache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py#L697).

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py) · **Recent**

Source documentation: Flash Attention v3 forward pass compatible interface for AMD Triton implementation. This function maps v3 parameters to the existing AMD Triton implementation.

Symbols: [`fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py#L24), [`fwd_combine`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py#L595), [`get_scheduler_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py#L620).

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py) · **Recent**

Source documentation: Utilities for Flash Attention Triton AMD backend. This module contains essential runtime utilities: - GPU architecture detection - Global configuration flags - Tensor shape/stride helpers - FP8 type detection

Symbols: [`GpuArch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L80), [`is_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L179), [`get_shape_from_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L227), [`get_stride_from_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L257), [`get_padded_headsize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L272), [`round_multiple`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L284), [`is_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L293), [`get_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L299), [`remap_xcd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py#L315).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/fusions/attn_res.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/attn_res.py) · **Recent**

Source documentation: AttnRes forward, ported from fla 0.5.2 attnres_fwd_kernel. Per token: RMS-normalize each of the L residual candidates, score each with a dot product against q * w, softmax over the candidate axis, and return the weighted sum (optionally output-RMSNorm'd). One program per row; the whole D stays in registers (BD = next_pow2(D)) and the candidate axis is streamed in BL tiles with an online softmax, so each candidate tile is read once.

Symbols: [`attnres_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/attn_res.py#L46).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py) · **Recent**

Source documentation: Pack per-channel fp4 (E2M1) V into the f4f4 kernel's col-major LDS operand layout: per 128-kv tile, 8 blocks of 1024 B (u = 2*n + k; n = head-dim 32-block 0..3, k = kv 64-half 0..1); each block is 64 kv-cols x 16 nibble-bytes (even chan -> low nibble, odd chan -> high nibble). One program packs one 1024 B block. Cosine-equivalent to the numpy packer; only exact E2M1 tie-midpoints may differ by one code (arbitrary, cosine-neutral).

Symbols: [`mha_v4_per_tensor_amax_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L16), [`mha_v4_per_tensor_scale_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L39), [`mha_v4_per_tensor_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L64), [`_compute_mx_quant_and_scale_rne`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L94), [`sage_quant_v_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L220), [`sage_quant_v_amax_partial_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L286), [`sage_quant_v_amax_finalize_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L333), [`_e2m1_code`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L359), [`_mxfp4_scale_from_amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L375), [`_e2m1_pack_native`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L389), [`sage_quant_v_fp4_colmajor_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L411), [`sage_quant_v_mxfp4_colmajor_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L489), [`_rotate_quantize_q_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L577), [`_rotate_quantize_k_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L713), [`_rotate_quantize_qk_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L839), [`_rot_q_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1025), [`_rot_k_only_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1134), [`_compute_delta_s_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1210), [`_q_smooth_int8_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1294), [`sage_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1380), [`_general_quant_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py#L1539).

Parameters referenced in this file: `BLOCK_D`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `BLOCK_SIZE`, `BLOCK_SIZE_OUT_DIM`, `BLOCK_SIZE_QUANT_DIM`, `BLOCK_SIZE_QUANT_MX_SCALE`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1101/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/fav3_sage/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/fav3_sage/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/na3d_flash/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/na3d_flash/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage_mxfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage_mxfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/na3d_flash/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/na3d_flash/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate_close_block/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate_close_block/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_sequence/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_sequence/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
- [op_tests/triton_tests/attention/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/__init__.py)
- [op_tests/triton_tests/attention/mha_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/mha_test_utils.py)
- [op_tests/triton_tests/attention/test_chunked_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunked_pa_prefill.py)
- [op_tests/triton_tests/attention/test_dsv4_indexer.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_dsv4_indexer.py)
- [op_tests/triton_tests/attention/test_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_extend_attention.py)
- [op_tests/triton_tests/attention/test_fav3_sage.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage.py)
- [op_tests/triton_tests/attention/test_fav3_sage_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage_compile.py)
- [op_tests/triton_tests/attention/test_la_paged.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_la_paged.py)
- [op_tests/triton_tests/attention/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha.py)
- [op_tests/triton_tests/attention/test_mha_dao_ai.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_dao_ai.py)
- [op_tests/triton_tests/attention/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_fp8.py)
- [op_tests/triton_tests/attention/test_mha_spatial_swizzle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_spatial_swizzle.py)
- [op_tests/triton_tests/attention/test_mha_v3.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_v3.py)
- [op_tests/triton_tests/attention/test_mha_with_pe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_pe.py)
- [op_tests/triton_tests/attention/test_mha_with_sink.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_sink.py)
- [op_tests/triton_tests/attention/test_na3d_flash.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_na3d_flash.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill.py)
- [op_tests/triton_tests/attention/test_prefill_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_prefill_attention.py)
- [op_tests/triton_tests/attention/test_qsa.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_qsa.py)
- [op_tests/triton_tests/fusions/test_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_attn_res.py)
- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/test_flydsl_fmha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fmha.py)
- [op_tests/test_fmha_fwd_mxfp8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_mxfp8_asm.py)
- [op_tests/test_fmha_fwd_with_sink_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_asm.py)
- [op_tests/test_fmha_fwd_with_sink_varlen_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_varlen_asm.py)
- [op_tests/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha.py)
- [op_tests/test_mha_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl.py)
- [op_tests/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_fp8.py)
- [op_tests/test_mha_native_splitkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_native_splitkv.py)
- [op_tests/test_mha_optional_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_optional_flydsl.py)
- [op_tests/test_mha_v4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4.py)
- [op_tests/test_mha_v4_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4_sparse.py)
- [op_tests/test_mha_varlen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen.py)
- [op_tests/test_mha_varlen_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_fp8.py)
- [op_tests/test_mha_varlen_large_kv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_large_kv.py)

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward reference with dropout disabled; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
