---
title: "Linear attention and gated delta rules: AITER flydsl"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Linear attention and gated delta rules — AITER flydsl

Prefill commonly uses chunked state updates while decode applies a recurrent update. State layout, gate transforms, convolution state and in-place update order must match the wrapper.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/linear_attention_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py) · **Recent**

Source documentation: High-level FlyDSL Linear Attention APIs.

Symbols: [`_mtp_variant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L37), [`_tile_warps`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L42), [`_decode_warp_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L89), [`_decode_tiling`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L114), [`get_default_kwargs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L143), [`_mtp_rung`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L211), [`_mtp_shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L236), [`_mtp_tiling`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L258), [`get_mtp_default_kwargs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L282), [`_mtp_kwargs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L288), [`flydsl_gdr_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L327), [`_is_supported_arch_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L455), [`_is_supported_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L463), [`_unit_strided`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L470), [`_on_device`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L481), [`_mtp_shapes_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L492), [`_flydsl_gdr_mtp_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L543), [`_flydsl_gdr_mtp_sglang_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L581), [`_snapshot_store_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L641), [`_require_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L647), [`_mtp_common_checks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L670), [`_mtp_stream`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L737), [`_mtp_launch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L744), [`_launch_flydsl_gdr_mtp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L857), [`flydsl_gdr_mtp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L898), [`flydsl_gdr_mtp_sglang`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py#L956).

Parameters referenced in this file: `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/linear_attention_prefill_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py) · **Recent**

Source documentation: FlyDSL host wrappers for GDN prefill. The module exposes hidden-state and prepare kernels. The prepare wrapper matches the Triton prepare layout and exponent-domain contract, so both paths can be selected independently through chunk_gated_delta_rule_opt_vk. Call gdn_prepare_flydsl_supported before selecting the fused prepare path.

Symbols: [`_hipeq_device_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L68), [`_hipeq_shared_memory_per_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L74), [`_hipeq_compute_bv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L92), [`_hipeq_select_bv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L108), [`_hipeq_varlen_host_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L123), [`_opt_check`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L151), [`_placeholder_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L159), [`_tuned_bv_runtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L170), [`_require_contiguous`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L179), [`_as_contiguous`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L185), [`_load_tuned_bv_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L189), [`reload_tuned_bv_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L240), [`_tuned_bv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L249), [`_as_int32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L293), [`_as_state_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L313), [`_resolve_prologue`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L322), [`_resolve_state_dtype`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L378), [`_resolve_snapshot_dtype`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L399), [`_get_or_compile_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L409), [`chunk_gated_delta_rule_fwd_h_flydsl_opt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L484), [`_device_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L921), [`_pad_grid_x_odd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L926), [`_is_cdna_mfma_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L932), [`gdn_prepare_flydsl_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L946), [`gdn_prepare_fwd_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py#L967).

### `aiter/ops/flydsl/kernels/gdr_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_common.py) · **Recent**

Source documentation: Vector-copy helpers shared by GDR decode and prefill kernels.

Symbols: [`_gview`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_common.py#L9), [`_load_vec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_common.py#L17), [`_store_vec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_common.py#L24).

### `aiter/ops/flydsl/kernels/gdr_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py) · **Recent**

Source documentation: Gated delta rule over a speculative draft window. Undoing a rejected token needs a rollback point and a per-token record; mode picks whose contract to follow. MTP_MODE_CHAIN is vLLM's: rollback at state_indices[n, num_accepted - 1], and every token checkpoints into state_indices[n, t], the last of which is the final store. MTP_MODE_SNAPSHOT is SGLang's: rollback at state_indices[n], record in intermediate_states_buffer.

Symbols: [`_gview64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py#L16), [`_fast_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py#L29), [`create_vk_gdr_decode_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py#L34), [`create_vk_gdr_mtp_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py#L573).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py) · **Recent**

Source documentation: Gated Delta Net K5 hidden-state recurrence kernel (@flyc.kernel API). HIP-aligned fork: same mfma_f32_16x16x16bf16_1k instruction and warp partition (BT split-M, K split across waves, V not split) as the hand-tuned HIP/C++ K5 kernel; writes the public [..., V, K] layout through a [V][K] transpose buffer + b128 store (fp32 snapshots use two half-BV LDS rounds).

Symbols: [`_make_fast_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py#L24), [`_make_bf16_converter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py#L37), [`compile_chunk_gated_delta_h`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py#L47).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py) · **Recent**

Source documentation: FlyDSL GDN prefill chunk-prepare kernel for the BT=64 path. For each chunk, form the strictly lower-triangular gated KKT matrix A and C = (I + A)^-1, then produce: g_cumsum : [B, H, T] in-chunk prefix sum of g w_bar : [B, H, T, K] C @ (k * beta * exp(g_cumsum)) u_bar : [B, H, T, V] C @ (v * beta) Inputs are token-major and outputs are head-major. g_scale controls whether g_cumsum is published in natural-log or log2 space.

Symbols: [`_exp2_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L30), [`_mfma16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L39), [`_mfma16_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L51), [`_acc16_n4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L67), [`_gemm16_n4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L74), [`_accum_to_bf16x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L86), [`_swizzled_vt_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L95), [`_load_mfma_tile_vt_tiled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L111), [`_tile16_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L121), [`_load_fp32_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L132), [`_load_fp32_tile_raw`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L146), [`_store_fp32_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L160), [`_identity_frag`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L169), [`_wy_prefetch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L177), [`_wy_epilogue_to_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L184), [`_wy_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L193), [`_wave_inclusive_scan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L206), [`compile_gdn_prepare`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py#L217).

Parameters referenced in this file: `BLOCK_THREADS`.

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
- [op_tests/test_flydsl_gdr_mtp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gdr_mtp.py)
- [op_tests/test_flydsl_linear_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention.py)
- [op_tests/test_flydsl_linear_attention_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention_prefill.py)
- [op_tests/test_gated_delta_rule.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gated_delta_rule.py)
- [op_tests/test_gdn_prepare.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdn_prepare.py)
- [op_tests/test_gdr_decode_packed_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdr_decode_packed_bf16.py)
- [op_tests/test_split_gdr_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_split_gdr_update.py)
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

## Performance and validation

Validate a sequence of updates against the reference, not just one isolated call. Include preparation, triangular solves and normalization for the chosen path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
