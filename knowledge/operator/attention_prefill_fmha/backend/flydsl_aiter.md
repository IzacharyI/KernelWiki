---
title: "Prefill and full attention: AITER flydsl"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER flydsl

Tiled attention interleaves QK products, online softmax and probability-value products. Causal/variable-length masks, dropout or backward outputs distinguish entry points with similar names.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/fmha_bwd_gfx942.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py) · **Recent**

Source documentation: Host side of the FlyDSL varlen FMHA backward (d_qk=192, d_v=128, causal, bf16, gfx942). Grid planning plus cached CompiledFunction dispatch for the kernels in kernels/fmha_bwd_gfx942/fmha_bwd_core.py: a k_delta pre-pass (D = rowsum(dO*O)) and one fused k_bwd that carries all five backward GEMMs. The device code stays under kernels/ and everything host-side lives here.

Symbols: [`_num_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py#L51), [`_interleave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py#L62), [`_split`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py#L70), [`_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py#L78), [`flash_attn_varlen_bwd_d192_gfx942`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_bwd_gfx942.py#L97).

### `aiter/ops/flydsl/fmha_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py) · **Recent**

Source documentation: High-level FlyDSL Flash Attention APIs. flydsl_flash_attn_batch_func / flydsl_flash_attn_varlen_func dispatch by arch and dtype: gfx950 fp8 to kernels/fmha_gfx950, gfx1250 bf16/f16 to the m32x8 prefill kernel, anything else None so the caller falls through to CK/Triton. flydsl_flash_attn_func (gfx1201 / RDNA4) wraps the flash_attn_func_gfx1201 kernel with: - Build cache keyed by (num_heads, head_dim, causal, dtype, waves_per_eu, daz).

Symbols: [`_torch_dtype_to_str`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L68), [`_get_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L77), [`flydsl_flash_attn_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L95), [`_check_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L226), [`_fp8_gfx950_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L240), [`flydsl_flash_attn_varlen_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L312), [`flydsl_flash_attn_batch_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L443), [`flydsl_flash_attn_varlen_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py#L556).

Parameters referenced in this file: `BLOCK_M`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py) · **Recent**

Source documentation: FlyDSL Flash Attention fp8 (e4m3fn) forward for gfx950. Q/K/V are pre-quantized e4m3fn with per-tensor fp32 shape-[1] descales; the output is bf16. Dense, packed-varlen and split-K.

Symbols: [`_is_valid_softmax_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L56), [`_fp8_rescale_threshold`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L61), [`_gpu_arch_cached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L69), [`_num_cu_cached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L77), [`_gpu_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L84), [`_num_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L88), [`_fp8_config_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L93), [`flydsl_flash_attn_fp8_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L120), [`_fp8_auto_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L152), [`_fp8_batch_interleave_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L165), [`_fp8_auto_kv_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L175), [`_build_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L227), [`flydsl_flash_attn_fp8_func`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py#L270).

Parameters referenced in this file: `BLOCK_M`.

### `aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py) · **Recent**

Source documentation: Flash Attention kernel for gfx1201 (RDNA4). Uses 16x16x16 wave32 WMMA, online softmax, pipelined V loads, and flattened BSHD inputs. Requires head_dim >= 64 and head_dim % 32 == 0.

Symbols: [`build_flash_attn_func_module_primary`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py#L27).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_N_OUT`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py) · **Recent**

Source documentation: FlyDSL FMHA backward for MLA-style d_qk=192 / d_v=128, varlen THD, causal, bf16, gfx942. The MFMA primitive used everywhere is v_mfma_f32_16x16x16_bf16_1k: C[16m, 16n] += A[16m, 16k] . B[16n, 16k]^T k_bwd carries TWO JOB TYPES on one grid.

Symbols: [`_mfma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L101), [`_exp2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L115), [`_zero4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L119), [`_elem`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L123), [`_pack4_trunc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L128), [`_to_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L143), [`_buffer_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L149), [`_atom`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L161), [`_ldv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L166), [`_ld1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L173), [`_stv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L180), [`_st1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L187), [`_bf16x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L194), [`_packn_rne`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L198), [`build`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py#L224).

### `aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py) · **Recent**

Source documentation: 16-bit (bf16/fp16) Q/K/V staging managers for the gfx1250 MHA kernels. Each manager owns the global -> LDS (async) -> VGPR (WMMA fragment) path for one 16-bit-element operand (Q, K or V): the LDS layout, the global->LDS copy schedule and the fragment read. They are self-contained — a caller passes only the configuration it already maintains (hdim, gqa_ratio, kv block width, wave count) through the constructor plus the runtime warp_idx/lane_idx.

Symbols: [`_assert_multiple`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L80), [`_as_bases`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L85), [`_lds_ptr_b128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L99), [`_async_load_to_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L107), [`ProducerCtx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L169), [`BufferOpDescriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L184), [`AsyncCopyDescriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L203), [`TdmCopyDescriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L222), [`_warp_band`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L237), [`_async_band_descriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L265), [`_pow2_segments`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L331), [`_tdm_load_views`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L343), [`_tdm_band_descriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L396), [`_AsyncTransport`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L441), [`_TdmTransport`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L477), [`QManager16bV1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L522), [`KManager16bV1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L787), [`VManager16bV1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L913), [`QManager16bV2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1020), [`KManager16bV2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1177), [`VManager16bV2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1184), [`OManager16bV1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1194), [`OManager16bV2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1419), [`OManager16bV3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py#L1601).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py) · **Recent**

Source documentation: MHA Forward Prefill kernel — m32x8 design, gfx1250 (MI400 / mi450). A clean FlyDSL kernel written in the high-level layout-algebra style (tiled copy / tiled MMA + SharedAllocator). m32x8 names the threadgroup shape: **8 waves per threadgroup**, each wave owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32, so a threadgroup is 8 * 32 = 256 threads and BLOCK_M = 32 * 8 = 256 Q rows.

Symbols: [`supported_num_q_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L100), [`supports_gqa_ratio`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L113), [`_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L122), [`_m_tile_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L126), [`WarpType`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L133), [`_warp_id`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L243), [`_lane_id`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L248), [`_copy_wait`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L255), [`_kv_drain_depths`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L263), [`_bare_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L268), [`_copy_fence`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L277), [`_load_seqlen_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L290), [`_load_sink_logit`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L298), [`_packed_tile_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L305), [`_wmma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L332), [`_convert_vectors`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L347), [`_keepalive`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L352), [`_ring_num_prefetch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L364), [`_ring_drive`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L374), [`_tree_reduce_multi`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L473), [`_softmax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L501), [`_pv_qk_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L720), [`_wg_kv_span`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L860), [`_alloc_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L885), [`_prologue_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L922), [`schedule_bank`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L1270), [`_o_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L1298), [`_o_unpack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L1310), [`_core_attention_multi_kv_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L1322), [`_core_attention_one_kv_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L1963), [`_o_manager`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2258), [`_store_o_lse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2281), [`_epi_state`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2394), [`_epi_state_zeros`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2404), [`_epilogue_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2420), [`_zero_fill_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2502), [`kv_split_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2581), [`pick_n_block`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2599), [`build_fmha_fwd_prefill_a16w16_m32x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2618), [`_pick_one_kv_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2950), [`_pick_num_q_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2956), [`_ensure_thd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L2970), [`_ensure_bshd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L3088), [`flash_attn_varlen_m32x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L3204), [`flash_attn_batch_m32x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py#L3373).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py) · **Recent**

Source documentation: gfx950 DUALWAVE_SWP FP8 flash attention.

Symbols: [`build_flash_attn_dualwave_swp_fp8_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py#L38).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py) · **Recent**

Source documentation: Split-K combine pass and workspace sizing.

Symbols: [`DualwaveSplitKCombineContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py#L22), [`DualwaveSplitKCombineHelper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py#L149), [`dualwave_splitk_workspace_elems`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py#L244).

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_epilog.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_epilog.py) · **Recent**

Source documentation: Epilogue: output packing and the dense / split-K partial stores.

Symbols: [`DualwaveFp8StoreHelper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_epilog.py#L22).

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_gemm.py) · **Recent**

Source documentation: The two GEMMs: QK^T and PV.

Symbols: [`DualwaveFp8GemmHelper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_gemm.py#L20).

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py) · **Recent**

Source documentation: LDS data movement: Q load, K/V global->LDS, LDS->VGPR.

Symbols: [`DualwaveFp8QLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py#L18), [`DualwaveFp8KvGmemToLdsLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py#L35), [`DualwaveFp8KvLdsToVgprLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py#L147).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_softmax.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_softmax.py) · **Recent**

Source documentation: Online softmax, with causal / seq-pad masking.

Symbols: [`DualwaveFp8SoftmaxHelper`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_softmax.py#L33).

Parameters referenced in this file: `BLOCK_N`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py) · **Recent**

Source documentation: Per-block forward pass: shared primitives, traits, and the kernel context.

Symbols: [`_waitcnt_vm_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L36), [`_s_setprio`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L41), [`_read_exec_i64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L45), [`_ds_read_tr8_b64_imm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L51), [`_bitcast_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L68), [`_bitcast_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L72), [`_attn_mask_vec2_imm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L76), [`_reduction_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L85), [`_anchor_scalar_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L94), [`_anchor_v_o`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L106), [`_score_pair_to_lists`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L128), [`_score_lists_to_vecs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L136), [`_reduce_score_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L144), [`_lane_pair_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L154), [`_score_pair_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L159), [`_score_pair_sum`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L166), [`_scale_sub_score_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L175), [`_exp2_score_slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L200), [`_scale_o_accs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L215), [`_causal_pair_thresholds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L221), [`_apply_dualwave_causal_mask_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L245), [`_cu_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L259), [`_make_ws_rsrc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L269), [`_p_headroom_log2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L276), [`_store_lse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L284), [`_buffer_load_128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L314), [`_buffer_load_lds_128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L323), [`_buffer_store_128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L333), [`_stagger_extra_barrier_if_one`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L344), [`DualwaveSwpFp8Traits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L352), [`_make_dualwave_swp_fp8_traits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L436), [`dualwave_fp8_dma_per_iter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L601), [`_init_dualwave_thread_mapping`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L614), [`_init_dualwave_q_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L667), [`DualwaveFp8KernelContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py#L676).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/flydsl/bench_mha_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_bwd.py)
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
- [op_tests/test_flydsl_fmha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fmha.py)
- [op_tests/test_fmha_fwd_mxfp8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_mxfp8_asm.py)
- [op_tests/test_fmha_fwd_with_sink_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_asm.py)
- [op_tests/test_fmha_fwd_with_sink_varlen_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_varlen_asm.py)
- [op_tests/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha.py)
- [op_tests/test_mha_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl.py)
- [op_tests/test_mha_flydsl_varlen_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl_varlen_bwd.py)
- [op_tests/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_fp8.py)
- [op_tests/test_mha_native_splitkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_native_splitkv.py)
- [op_tests/test_mha_optional_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_optional_flydsl.py)
- [op_tests/test_mha_v4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4.py)
- [op_tests/test_mha_v4_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4_sparse.py)
- [op_tests/test_mha_varlen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen.py)
- [op_tests/test_mha_varlen_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_fp8.py)
- [op_tests/test_mha_varlen_large_kv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_large_kv.py)
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

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward or backward reference; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
