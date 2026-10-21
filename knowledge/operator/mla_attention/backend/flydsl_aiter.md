---
title: "Multi-head latent attention: AITER flydsl"
kind: implementation_knowledge
operator: mla_attention
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER flydsl

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/gather_kv_b_proj.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py) · **Recent**

Source documentation: FlyDSL backend for the MLA fused gather + kv_b_proj expansion. Supported: fp8 KV cache (OCP e4m3), fp8 weight in either row-major or shuffle_weight((16,16)) layout, per-output-row *or* 128x128 block weight scale, per-tensor activation scale, page_size 1, bf16 or scaled fp8 outputs on gfx950. The gfx1250 128+128 ptpc specialization is dispatched separately.

Symbols: [`lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L40), [`_lds_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L45), [`block_n_of`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L50), [`_default_head_tiles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L55), [`_cu_count`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L62), [`_default_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L67), [`_config_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L75), [`_raise`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L131), [`_validate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L135), [`_is_per_row_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L142), [`_unsupported_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L148), [`gather_kv_b_proj_flydsl_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L276), [`_output_scale_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L287), [`gather_kv_b_proj_flydsl_fp8_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L303), [`compile_gather_kv_b_proj`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L344), [`_arch_of`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L384), [`_as_i8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L390), [`_default_xcd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L398), [`gather_kv_b_proj_flydsl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py#L415).

Parameters referenced in this file: `BLOCK_M`, `waves_per_eu`.

### `aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py) · **Recent**

Source documentation: gfx1250 MLA gather + FlyDSL ptpc projection.

Symbols: [`_projection_runtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py#L18), [`_cluster_n`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py#L30), [`_run_projection`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py#L35), [`unsupported_reason`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py#L68), [`gather_kv_b_proj_flydsl_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py#L121).

### `aiter/ops/flydsl/mla_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py) · **Recent**

Source documentation: BF16 Q/KV form of :func:flydsl_mla_pagesize1_fp8_fp8; no scales.

Symbols: [`_require`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L23), [`_require_runtime`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L28), [`_require_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L33), [`_validate_pagesize1_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L48), [`_run_pagesize1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L123), [`flydsl_mla_pagesize1_fp8_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L247), [`flydsl_mla_pagesize1_bf16_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L301), [`_validate_pagesize64_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L354), [`flydsl_mla_pagesize64_fp8_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L395), [`flydsl_mla_decode_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py#L455).

### `aiter/ops/flydsl/mla_reduce_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py) · **Recent**

Source documentation: High-level FlyDSL MLA decode reduce API. Drop-in alternative for the HIP aiter.mla_reduce_v1 (csrc/kernels/mla/reduce.cu): same signature and in/out contract. Opt-in via AITER_MLA_REDUCE_FLYDSL=1; production keeps the HIP kernel by default. Kernels are compiled per (H, Dv, out_dtype, tier, output_lse) and lru_cache-backed; Tier.ALL (one kernel, device-side per-tile tier selection) mirrors HIP.

Symbols: [`_out_dtype_str`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L52), [`_pointer_arg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L62), [`_require_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L75), [`_validate_actual_max_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L98), [`_validate_mla_reduce_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L112), [`_resolve_actual_max_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L251), [`flydsl_mla_reduce_v1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py#L272).

Parameters referenced in this file: `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gather_kv_b_proj_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_kv_b_proj_gfx1250.py) · **Recent**

Source documentation: MLA gather projection WMMA kernel for gfx1250. This is deliberately not a generic GEMM entry point. It gathers page-size-1 cache rows directly into the projection pipeline's padded LDS layout, multiplies the 512-wide FP8 latent by a ptpc-preshuffled FP8 weight, and writes BF16 128-wide K-nope, 64-wide RoPE, and 128-wide V head slices.

Symbols: [`launch_gather_kv_b_proj_8wave_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_kv_b_proj_gfx1250.py#L53).

Parameters referenced in this file: `num_buffers`, `num_warps`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py) · **Recent**

Source documentation: Helpers shared by the gfx1250 MLA kernels.

Symbols: [`_instruction_prefetch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L14), [`_xor16_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L30), [`_concat_wmma_operand`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L37), [`_concat_wmma_operand_k64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L43), [`_concat_ds_tr8_b64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L47), [`_rmem_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L53), [`make_fp8_wmma_atom`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L59), [`_pack_fp8x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L71), [`_dwordx4_iter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L90), [`make_global_load_b128`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L97), [`wave_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L126), [`_unhoisted`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L156), [`_softmax_all`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L170), [`_round_robin`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L201), [`_drain`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L211), [`_min_u32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L219), [`_wait_scalars`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py#L225).

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py) · **Recent**

Source documentation: gfx1250 page-size-1 BF16 persistent MLA decode stage-1 kernel, four waves.

Symbols: [`keep_alive`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py#L63), [`kv_row_offset`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py#L114), [`_softmax_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py#L123), [`launch_mla_pagesize1_bf16_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py#L231).

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py) · **Recent**

Source documentation: gfx1250 page-size-1 FP8 persistent MLA decode stage-1 kernel, four waves.

Symbols: [`kv_row_offset`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py#L96), [`_softmax_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py#L105), [`launch_mla_pagesize1_fp8_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py#L233).

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py) · **Recent**

Source documentation: gfx1250 MLA page-size-64 FP8 stage-1 kernel.

Symbols: [`_dword_iter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py#L69), [`_rope_row_to_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py#L76), [`launch_mla_pagesize64_fp8_fp8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py#L91).

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`.

### `aiter/ops/flydsl/kernels/mla_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py) · **Recent**

Source documentation: FlyDSL MLA decode reduce/combine epilogue. Stage-2 of split-KV MLA decode: merges per-split partial outputs O_i (fp32) weighted by exp(LSE_i - LSE_max) (online softmax) into the final output (bf16/fp16), and optionally the merged LSE.

Symbols: [`_out_numeric_t`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L64), [`_vector_elements`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L73), [`_make_vector`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L79), [`_pointer_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L84), [`_pointer_buffer_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L90), [`_f32_copy_atom`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L95), [`_chunk_tile_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L115), [`_load_partial_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L122), [`_out_copy_atom`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L145), [`_store_final_out`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L165), [`_store_lse`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L186), [`_exp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L190), [`_is_zero_or_nan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L195), [`Tier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L202), [`select_tier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L227), [`should_use_persistent_launch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L237), [`compile_mla_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L251), [`compile_mla_decode_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L917), [`plan_splitk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L1057), [`derive_actual_max_splits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L1092), [`plan_splitk_capture_safe`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L1104), [`_get_splitk_scratch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L1167), [`compile_mla_reduce_splitk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py#L1184).

Parameters referenced in this file: `waves_per_eu`.

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

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
