---
title: "Multi-head latent attention: AITER hip"
kind: implementation_knowledge
operator: mla_attention
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER hip

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/mla_metadata.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_metadata.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_v4_prefill_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h) · **Recent**

Symbols: [`ceil_div`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L513), [`opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L517), [`opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L519), [`opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L521), [`opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L523), [`opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L528), [`opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L532), [`opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L536), [`opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L540), [`global_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L558), [`global_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L601), [`global_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L616), [`sched_compute_qk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L633), [`make_layout_q`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L658), [`make_layout_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L679), [`make_layout_gkv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L701), [`make_layout_skv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L723), [`make_layout_rk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L741), [`make_layout_rv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L766), [`make_layout_kv_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L804), [`attn_row_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L822), [`attn_row_scale_sub`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L838), [`attn_exp2_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L846), [`attn_row_sum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L854), [`scale_output_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L870), [`attn_mask_vec2_imm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L876), [`attn_mask_oob_score`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L892), [`attn_mask_oob_value`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L937), [`mla_v4_prefill_accum_le2_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L960), [`mla_v4_prefill_accum_pipelined`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1083), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1648), [`make_layout_q`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1748), [`make_layout_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1768), [`make_layout_gkv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1790), [`make_layout_skv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1810), [`make_layout_rk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1828), [`make_layout_rv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1855), [`make_layout_kv_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1901), [`attn_row_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1920), [`attn_row_scale_sub`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1947), [`attn_exp2_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1955), [`attn_row_sum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1963), [`scale_output_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1991), [`attn_mask_oob_score`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L1997), [`attn_mask_oob_value`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2028), [`mla_v4_prefill_accum_pipelined`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2048), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2146), [`sched_compute_qk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2286), [`make_layout_q_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2309), [`make_layout_q_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2330), [`make_layout_q_mxscl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2350), [`make_layout_kv_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2370), [`make_layout_gk_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2388), [`make_layout_sk_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2409), [`make_layout_rk_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2426), [`make_layout_rk_mxscl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2448), [`make_layout_gk_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2476), [`make_layout_sk_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2495), [`make_layout_cp_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2512), [`make_layout_rk_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2529), [`make_layout_rv_dequant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2553), [`make_layout_sv_dequant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2580), [`make_layout_rv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2612), [`make_layout_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2650), [`scale_output_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2671), [`attn_row_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2677), [`attn_row_scale_sub`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2691), [`attn_exp2_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2699), [`attn_row_sum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2707), [`attn_mask_vec2_imm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2721), [`attn_mask_oob_score`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2737), [`attn_mask_oob_value`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2782), [`mla_v4_prefill_accum_le2_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L2805), [`mla_v4_prefill_accum_pipelined`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3004), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3701), [`make_layout_q_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3846), [`make_layout_q_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3866), [`make_layout_q_mxscl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3885), [`make_layout_rk_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3904), [`make_layout_rk_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3921), [`make_layout_sk_nope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3937), [`make_layout_sk_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3956), [`make_layout_rv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L3975), [`make_layout_kv_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4010), [`make_layout_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4027), [`attn_row_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4048), [`attn_row_scale_sub`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4075), [`attn_exp2_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4083), [`attn_row_sum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4091), [`scale_output_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4119), [`attn_mask_oob_score`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4125), [`attn_mask_oob_value`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4156), [`reorder_mxscl_for_opsel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4177), [`mla_v4_prefill_accum_pipelined`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4195), [`__launch_bounds__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h#L4348).

### `csrc/kernels/mla/metadata.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/metadata.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh10_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh10_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh112_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh112_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh128_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh128_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh128_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh128_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh12_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh12_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh16_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh16_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh16_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh16_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh1_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh1_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh24_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh24_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh2_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh2_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh32_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh32_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh32_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh32_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh40_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh40_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh48_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh48_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh48_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh48_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh4_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh4_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd64.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd64.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh80_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh80_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh8_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh8_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh8_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh8_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh96_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh96_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh96_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh96_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_hk.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_hk.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_reduce.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_reduce.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/hk_v32_decode_fwd.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/hk_v32_decode_fwd.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/hk_v40_decode_fwd.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/hk_v40_decode_fwd.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode_rope.py)
- [op_tests/test_concat_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla.py)
- [op_tests/test_concat_cache_mla_seg.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla_seg.py)
- [op_tests/test_flydsl_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gather_kv_b_proj.py)
- [op_tests/test_flydsl_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mla_reduce.py)
- [op_tests/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla.py)
- [op_tests/test_mla_decode_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_gate.py)
- [op_tests/test_mla_decode_pagesize64.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_pagesize64.py)
- [op_tests/test_mla_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_flydsl.py)
- [op_tests/test_mla_ltx.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_ltx.py)
- [op_tests/test_mla_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent.py)
- [op_tests/test_mla_persistent_round_robin.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent_round_robin.py)
- [op_tests/test_mla_prefill_ps.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_prefill_ps.py)
- [op_tests/test_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_reduce.py)
- [op_tests/test_mla_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_sparse.py)
- [op_tests/test_mla_stage2_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_stage2_merge.py)
- [op_tests/test_mla_v40_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v40_persistent.py)
- [op_tests/test_mla_v4_kargpreld.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_kargpreld.py)
- [op_tests/test_mla_v4_nm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_nm.py)
- [op_tests/triton_tests/attention/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla.py)
- [op_tests/triton_tests/attention/test_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_decode_rope.py)
- [op_tests/triton_tests/attention/test_mla_v4_triton.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_v4_triton.py)
- [op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py)
- [op_tests/triton_tests/test_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_gather_kv_b_proj.py)
- [op_tests/triton_tests/utils/mla_decode_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_decode_ref.py)
- [op_tests/triton_tests/utils/mla_extend_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_extend_ref.py)

## Performance and validation

Record cache layout, head counts, latent/rope dimensions and query length. Compare the complete wrapper and its reduction/projection helpers.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
