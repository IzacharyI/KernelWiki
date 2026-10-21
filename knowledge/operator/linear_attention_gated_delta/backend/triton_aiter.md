---
title: "Linear attention and gated delta rules: AITER triton"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Linear attention and gated delta rules — AITER triton

Prefill commonly uses chunked state updates while decode applies a recurrent update. State layout, gate transforms, convolution state and in-place update order must match the wrapper.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/chunk_kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py) · **Recent**

Source documentation: Chunked Kimi Delta Attention prefill, Gluon on gfx1250 and gfx950: prepare, then the walk. Runs on gluon for 950/1250, takes the triton path for 942

Symbols: [`prepare_chunk_kda_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L28), [`_empty`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L46), [`_check_tokens`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L52), [`_check_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L57), [`chunk_kda_prepare`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L63), [`chunk_kda_walk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L148), [`_walk_gfx950`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L293), [`chunk_kda`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py#L356).

Parameters referenced in this file: `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/kda.py) · **Recent**

Source documentation: Fused recurrent Kimi Delta Attention (KDA), gfx950/gfx1250 Gluon decode path. Per token: S = exp(g_t) * S, then S += beta_t * k_t (x) (v_t - S^T k_t), then o_t = S^T q_t. Requires triton >= 3.6.0 on gfx1250. Args: q: [B, T, H, K] queries k: [B, T, H, K] keys v: [B, T, HV, V] values g: [B, T, HV, K] log-space decay gates beta: [B, T, HV] scalar or [B, T, HV, V] headwise write strengths. A_log: [HV] gate parameter dt_bias: [HV, K] optional bias added to g inside the gate chain.

Symbols: [`fused_recurrent_kda`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/kda.py#L31), [`fused_recurrent_kda_packed_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/kda.py#L395).

Parameters referenced in this file: `num_buffers`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py) · **Recent**

Source documentation: Fused Qwen3-Next Gated DeltaNet decode, with an optional FP8 quant epilogue. One Gluon kernel launch replaces the four-kernel decode chain fused_qkvzba_split_reshape_cat -> causal_conv1d_update -> fused_recurrent_gated_delta_rule_packed_decode -> layer_norm_gated and can additionally emit the per-head group-128 FP8 activations that a block-FP8 out_proj consumes directly, removing a separate quantization launch.

Symbols: [`_arch_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py#L52), [`_gluon_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py#L73), [`fused_gdn_decode_qkvz_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py#L99), [`fused_gdn_decode_qkvz`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py#L232).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py) · **Recent**

Source documentation: Fused Qwen3-Next Gated DeltaNet *prefill*, with an FP8 group-quant epilogue.

Symbols: [`_arch_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L95), [`_gluon_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L116), [`_select_tile_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L141), [`FP8PrecisionConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L171), [`_run_chunked_delta`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L188), [`gdn_prefill_group_fp8_quant_m1024_3071`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L363), [`_Schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L456), [`_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L477), [`gdn_prefill_group_fp8_quant_m12289_16384_b1_5`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L537), [`gdn_prefill_group_fp8_quant_m12289_16384_b6_15`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L841), [`gdn_prefill_group_fp8_quant_m3072_16384`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L1109), [`_load_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L1412), [`fused_gdn_prefill_qkvz_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L1429), [`fused_gdn_prefill_qkvz`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py#L1625).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_kda_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode.py) · **Recent**

Source documentation: Fused KDA decode: conv1d + delta-rule recurrence + gated RMSNorm. Single Triton kernel launch. Fuses three separate operations into one, eliminating q/k/v intermediate HBM traffic and kernel launch overhead.

Symbols: [`_launch_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode.py#L28), [`fused_kda_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode.py#L39).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py) · **Recent**

Source documentation: Unified fused KDA decode: conv1d + recurrence + gated RMSNorm. Python wrapper for the unified kernel. Mode is auto-selected from args: - state_indices only → normal decode - state_indices + num_accepted → DSpark spec decode - slot_idx + buf_k/u/g → ReplaySSM - slot_idx + buf_k/u/g + conv_state_indices → DSpark + ReplaySSM

Symbols: [`fused_kda_decode_unified`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py#L24).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py) · **Recent**

Source documentation: Fused sigmoid-gated delta rule over packed QKV. draft_window is host metadata required for FlyDSL dispatch. FlyDSL falls back to Triton during CUDA Graph capture.

Symbols: [`_flydsl_gdr_enabled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py#L20), [`_try_flydsl_mtp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py#L29), [`fused_rearrange_sigmoid_gated_delta_rule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py#L186).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/gated_delta_rule.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py) · **Recent**

Source documentation: Gated Delta Rule Operations (Forward Only). This module provides high-level interfaces for gated delta rule computations, including both fused recurrent and chunk-based implementations. Important Note: Only forward pass is implemented in aiter. These functions do NOT support gradient computation or backward pass. For training with autograd, please use the flash-linear-attention library instead. These implementations are optimized for inference and forward-only operations.

Symbols: [`fused_recurrent_gated_delta_rule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py#L39), [`chunk_gated_delta_rule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py#L211), [`chunk_gated_delta_rule_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py#L354), [`chunk_gated_delta_rule_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py#L460).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py) · **Recent**

Source documentation: Kimi Delta Attention (KDA) chunked forward pass (Forward Only). This is the public entry point for the chunk_delta_attn Triton kernels. The signature mirrors fla.ops.kda.chunk_kda so serving stacks can swap the two backends without reshaping tensors or re-deriving the gate. Important Note: Only the forward pass is implemented. These functions do NOT support gradient computation. For training, use the flash-linear-attention library.

Symbols: [`chunk_kimi_delta_attn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py#L44).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py) · **Recent**

Source documentation: Fused KDA decode kernel: conv1d + delta-rule recurrence + gated RMSNorm. Single Triton kernel. Grid: (batch, heads).

Symbols: [`_valid_spec_prefix`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py#L25), [`fused_conv_recurrent_norm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py#L48), [`fused_kda_spec_parallel_v_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py#L444), [`fused_kda_spec_finalize_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py#L667).

Parameters referenced in this file: `num_stages`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm_unified.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm_unified.py) · **Recent**

Source documentation: Unified fused KDA decode kernel: conv1d + recurrence + gated RMSNorm.

Symbols: [`_fused_kda_decode_unified_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm_unified.py#L25).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_rearrange_sigmoid_gdr.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_rearrange_sigmoid_gdr.py) · **Recent**

Symbols: [`fused_rearrange_sigmoid_gated_delta_rule_update_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_rearrange_sigmoid_gdr.py#L24).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_recurrent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_recurrent.py) · **Recent**

Source documentation: Fused recurrent gated delta rule forward kernel (Forward only). This module provides an optimized fused recurrent implementation of the gated delta rule. Note: Only forward pass is implemented. Backward pass is not supported in aiter.

Symbols: [`_fused_recurrent_gated_delta_rule_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_recurrent.py#L47).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py) · **Recent**

Source documentation: Fused kernel that combines sigmoid gating computation with recurrent delta rule update.

Symbols: [`is_cuda`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L12), [`get_cuda_autotune_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L16), [`get_hip_autotune_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L22), [`get_autotune_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L28), [`fused_sigmoid_gating_delta_rule_update_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L47), [`fused_sigmoid_gating_delta_rule_update`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py#L196).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py) · **Recent**

Source documentation: Fused kernels for QKVZBA split, reshape and concatenation.

Symbols: [`_fused_qkvzba_split_reshape_cat_decode_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L9), [`fused_qkvzba_split_reshape_cat_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L98), [`_fused_qkvzba_prefill_kernel_nqk1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L144), [`_fused_qkvzba_prefill_kernel_nqk2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L191), [`_fused_qkvzba_prefill_kernel_nqk4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L261), [`_fused_qkvzba_prefill_kernel_nqk8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L324), [`_fused_qkvzba_split_reshape_cat_prefill_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L387), [`fused_qkvzba_split_reshape_cat_prefill`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py#L472).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py) · **Recent**

Source documentation: Checks the current operating system, Triton version, and Python version, issuing warnings if they don't meet recommendations. This function's body only runs once due to lru_cache.

Symbols: [`check_environments`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L62), [`get_abs_err`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L99), [`get_err_ratio`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L103), [`assert_close`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L109), [`tensor_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L123), [`input_guard`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L166), [`require_version`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L208), [`Action`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L236), [`deprecate_kwarg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L243), [`checkpoint`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L397), [`check_pytorch_version`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L405), [`_cpu_device_warning`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L409), [`get_multiprocessor_count`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L416), [`get_available_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L432), [`map_triton_backend_to_torch_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L440), [`get_all_max_shared_mem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L493), [`Backend`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L506), [`check_shared_mem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L521), [`_register_aliases`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py#L549).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py) · **Recent**

Source documentation: Chunk-based gated delta rule forward computation. This module implements the chunk-based parallel computation for the gated delta rule. Note: Only forward pass is implemented. Backward pass is not supported in aiter.

Symbols: [`_is_unsupported_gfx12_runtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py#L45), [`_is_gfx12_runtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py#L55), [`chunk_gated_delta_rule_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py#L64), [`chunk_gated_delta_rule_fwd_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py#L151), [`chunk_gated_delta_rule_fwd_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py#L238).

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py) · **Recent**

Source documentation: Chunk-based hidden state computation for gated delta rule (Forward only). This module computes the per-chunk hidden states and the recomputed value tensor (v_new) consumed by the chunked gated delta rule, supporting both the [K, V] and the transposed [V, K] hidden-state layouts.

Symbols: [`_gate_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L39), [`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L68), [`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L335), [`chunk_gated_delta_rule_fwd_h`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L568), [`chunk_gated_delta_rule_fwd_kernel_h_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L675), [`chunk_gated_delta_rule_fwd_h_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L891), [`chunk_gated_delta_rule_fwd_kernel_h_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L989), [`chunk_gated_delta_rule_fwd_h_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L1236), [`chunk_gated_delta_rule_bwd_dhu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py#L1421).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py) · **Recent**

Source documentation: Chunk-based output computation (Forward only). This module provides functions for computing the final output in chunk mode.

Symbols: [`_bp_ld1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L37), [`_bp_st1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L43), [`_bp_ld2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L49), [`_bp_st2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L60), [`chunk_fwd_kernel_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L90), [`chunk_bwd_kernel_dqkwg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L209), [`chunk_bwd_kernel_dv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L409), [`chunk_bwd_kernel_dv_local`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L512), [`chunk_fwd_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L585), [`chunk_fwd_kernel_o_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L653), [`chunk_fwd_o_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L733), [`chunk_fwd_kernel_o_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L826), [`chunk_fwd_o_opt_vk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L923), [`chunk_bwd_dv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L1032), [`chunk_bwd_dv_local`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L1086), [`chunk_bwd_dqkwg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py#L1137).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py) · **Recent**

Source documentation: Fused cumsum + KKT. Args: g: [B, T, H] k: [B, T, Hg, K] beta: [B, T, H] Returns: g_cumsum: [B, H, T] A: [B, T, H, chunk_size], strictly lower triangular

Symbols: [`safe_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py#L20), [`_fused_cumsum_kkt_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py#L26), [`fused_cumsum_kkt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py#L93), [`fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py#L175), [`fused_chunk_local_cumsum_scaled_dot_kkt_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py#L267).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py) · **Recent**

Source documentation: Fused sigmoid + gdn_gating kernel for prefill.

Symbols: [`fused_gdn_gating_sigmoid_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py#L9), [`fused_gdn_gating_and_sigmoid`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py#L50).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py) · **Recent**

Source documentation: Fused triangular solve + recompute w, u in a single kernel. Eliminates the intermediate Ai tensor (64x64 per chunk x head) global memory round-trip by keeping the inverse blocks in registers.

Symbols: [`_should_use_split_path`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L51), [`_bp_ld1d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L96), [`_bp_ld2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L102), [`_bp_st2d`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L113), [`fused_solve_tril_recompute_w_u_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L136), [`recompute_w_u_head_major_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L498), [`_run_split_path`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L590), [`fused_solve_tril_recompute_w_u`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py#L648).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py) · **Recent**

Source documentation: Shared utilities for chunk_delta_attn kernels.

Symbols: [`chunk_delta_attn_tuned_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L35), [`chunk_delta_attn_tuned_config_shortlist`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L49), [`_same_arg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L76), [`_same_call`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L82), [`tensor_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L91), [`_get_available_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L113), [`check_shared_mem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L128), [`softplus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L176), [`input_guard`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py#L188).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py) · **Recent**

Source documentation: Top-level forward function for chunk_delta_attn. Pipeline: 1. Gate cumsum: apply fused A_log / dt_bias / softplus (or sigmoid) gating and chunk-local prefix sum to produce gk (in log2 space). 2. Intra-chunk: compute Aqk, Akk inverse, and auxiliary W/U/KG tensors. 3. Inter-chunk: update recurrent hidden state H via chunk_gated_delta_rule_fwd_h. 4. Output: compute final output via chunk_gla_fwd_o (gated q * h + A * v).

Symbols: [`chunk_delta_attn_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py#L50).

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py) · **Recent**

Source documentation: Launch a Triton kernel without redoing the part that only depends on shapes. JITFunction.run binds the arguments, specializes each one, builds a cache key and looks the kernel up, and it does all of that on every call.

Symbols: [`bypassed`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py#L43), [`recording`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py#L54), [`_tensor_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py#L64), [`_FastLaunch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py#L72), [`fast_launch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py#L176).

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py) · **Recent**

Source documentation: Two-kernel fused KDA forward, following the FlashKDA kernel split. The default chunk_delta_attn_fwd pipeline runs five kernels (l2norm, beta sigmoid, gate cumsum, intra, delta_h, gla_o) and materializes g_cumsum, Aqk, Akk, w, u and h in HBM between them.

Symbols: [`_flash_kda_prepare_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L141), [`_seg_occupancy_class`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L372), [`_flash_kda_segment_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L414), [`_flash_kda_seg_scan_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L637), [`flash_kda_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L722), [`_gluon_k1_usable`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L752), [`_gluon_k2_usable`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L756), [`_log_route`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L766), [`_k2_gluon_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L789), [`_num_cus`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L814), [`_scan_bv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L828), [`_choose_chunks_per_seg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L835), [`_seq_bounds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L870), [`_build_segments`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L882), [`_check_paged_state_cache`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L934), [`flash_kda_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py#L968).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py) · **Recent**

Source documentation: Gate activation kernels for chunk_delta_attn. Provides: - beta_sigmoid_fwd: elementwise sigmoid for the beta gate (forward only). - chunk_delta_attn_gate_fwd: per-token fused gate (-exp(A)*softplus or lower_bound*sigmoid(exp(A)*g)) without cumsum (forward only).

Symbols: [`beta_sigmoid_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py#L34), [`beta_sigmoid_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py#L49), [`chunk_delta_attn_gate_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py#L84), [`chunk_delta_attn_gate_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py#L125).

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py) · **Recent**

Source documentation: GLA output computation for chunk_delta_attn forward pass. chunk_gla_fwd_o computes the final attention output: o = (q * exp2(gk)) @ h * scale + A @ v_new where h is the recurrent hidden state and A is the local intra-chunk attention matrix (Aqk).

Symbols: [`chunk_gla_fwd_kernel_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py#L63), [`chunk_gla_fwd_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py#L168).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py) · **Recent**

Source documentation: Intra-chunk attention kernels for chunk_delta_attn forward pass. Three-step pipeline: 1. Token-parallel kernel: computes diagonal Akk blocks + full Aqk row-by-row. 2. Sub-chunk intra kernel (safe-gate path): alternative diagonal-block kernel. 3. Fused inter + solve_tril kernel: computes off-diagonal Akk, then solves the full chunk triangular system in one pass → writes Akk_inv.

Symbols: [`chunk_delta_attn_fwd_kernel_intra_token_parallel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py#L74), [`_chunk_delta_attn_fwd_intra_token_parallel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py#L182), [`chunk_delta_attn_fwd_kernel_intra_sub_chunk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py#L242), [`chunk_delta_attn_fwd_kernel_inter_solve_fused`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py#L386), [`chunk_delta_attn_fwd_intra`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py#L688).

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py) · **Recent**

Source documentation: Recompute W and U tensors for chunk_delta_attn forward pass. recompute_w_u_fwd takes the chunk Akk inverse matrix (A), key (k), value (v), gate-cumsum (gk) and beta scalar to produce: - w = A @ (k * beta * exp2(gk)) [chunk-gated key correction] - u = A @ (v * beta) [chunk-gated value target] - qg = q * exp2(gk) [gated query, optional] - kg = k * exp2(gk_last - gk) [boundary-normalised key]

Symbols: [`recompute_w_u_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py#L55), [`recompute_w_u_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py#L170).

Parameters referenced in this file: `num_stages`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx950/triton/attention/fused_kda_decode/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fused_kda_decode/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_chunk_kda.py)
- [op_tests/op_benchmarks/triton/bench_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_flash_kda.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_kda_decode.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py)
- [op_tests/op_benchmarks/triton/bench_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_kda.py)
- [op_tests/triton_tests/attention/test_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunk_kda.py)
- [op_tests/triton_tests/attention/test_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_kda.py)
- [op_tests/triton_tests/gated_delta_net/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/__init__.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py)
- [op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py)
- [op_tests/triton_tests/kimi_delta_attn/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/__init__.py)
- [op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py)
- [op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py)
- [op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py)
- [op_tests/triton_tests/utils/gdr_mtp_refs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/gdr_mtp_refs.py)
- [op_tests/triton_tests/utils/kda_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/kda_ref.py)
- [op_tests/test_flydsl_gdr_mtp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gdr_mtp.py)
- [op_tests/test_flydsl_linear_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention.py)
- [op_tests/test_flydsl_linear_attention_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention_prefill.py)
- [op_tests/test_gated_delta_rule.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gated_delta_rule.py)
- [op_tests/test_gdn_prepare.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdn_prepare.py)
- [op_tests/test_gdr_decode_packed_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdr_decode_packed_bf16.py)
- [op_tests/test_split_gdr_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_split_gdr_update.py)

## Performance and validation

Validate a sequence of updates against the reference, not just one isolated call. Include preparation, triangular solves and normalization for the chosen path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
