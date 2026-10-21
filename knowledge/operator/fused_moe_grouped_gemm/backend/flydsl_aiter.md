---
title: "Fused MoE compute and communication: AITER flydsl"
kind: implementation_knowledge
operator: fused_moe_grouped_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused MoE compute and communication — AITER flydsl

These paths fuse several stages of MoE, sometimes including inter-GPU communication. Their synchronization and buffer ownership are part of correctness, alongside local GEMM arithmetic.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/comm_fused_moe_host.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py) · **Recent**

Source documentation: Production host runtime for communication-fused FlyDSL MoE. Set AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK=1 before runner creation to temporarily replay Window AR with dynamic MXFP8 partial and reduced payloads.

Symbols: [`ShapeKey`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L59), [`_gpu_model_from_pci`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L108), [`get_gpu_model`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L134), [`_window_mxfp8_fallback_enabled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L145), [`_mori_communicator`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L157), [`is_flydsl_comm_fused_moe_available`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L164), [`_int_value`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L178), [`_mega_defaults`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L186), [`config_name`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L197), [`config_record`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L297), [`_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L313), [`_check_block_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L340), [`_optional_float`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L349), [`_select_candidate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L357), [`_comm_configs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L372), [`_winner_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L386), [`winners_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L446), [`_align_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L462), [`_symmetric_allocation_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L466), [`_symmetric`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L473), [`_packed_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L478), [`_packed_symmetric`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L488), [`_window_workspace_sizes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L510), [`_runner_workspace_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L518), [`_WindowWorkspaceLayout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L527), [`_optional_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L595), [`_WindowPhaseView`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L602), [`_create_communicator`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L631), [`_register`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L639), [`_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L648), [`_barrier_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L658), [`_bind_inter_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L680), [`_stage2_args`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L688), [`_clear_missing_routes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L716), [`_RuntimeLayoutRunner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L750), [`_MegakernelRunner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L817), [`_WindowRunner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L956), [`_DirectRunner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L1329), [`create_runner`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L1484), [`_LazyRunners`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L1500), [`create_flydsl_comm_fused_runners`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py#L1607).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/mega_moe_tp.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_moe_tp.py) · **Recent**

Source documentation: Tensor-parallel MoE layer as one kernel (MegaMoE TP), gfx950. Experts are replicated over the TP group and inter_dim is sharded; one launch runs the collectives around GEMM1 + activation + GEMM2 of this rank's inter slice. comm_mode: "ag_rs" (sequence-parallel in/out) or "ar" (replicated in, all-reduced out). Weights are MXFP4 (shuffle_weight(16, 16) + e8m0_shuffle).

Symbols: [`mega_moe_tp_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_moe_tp.py#L87), [`MegaMoeTPConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_moe_tp.py#L97), [`MegaMoeTP`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_moe_tp.py#L137).

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py) · **Recent**

Source documentation: Communication and synchronization primitives for comm-fused MoE.

Symbols: [`peer_base`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L20), [`spin_until_epoch_i32_system`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L25), [`spin_until_epoch_i32_agent`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L38), [`buffer_tensor_from_addr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L50), [`load_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L59), [`store_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L73), [`load_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L88), [`store_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L110), [`compile_epoch_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L136), [`compile_epoch_barrier_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L182), [`e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L266), [`pack_fp8_words`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L276), [`mxfp4_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L299), [`pack_mxfp4_8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L312), [`_pack_mxfp4_32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L330), [`_unpack_mxfp4_32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L351), [`quantize_group32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L369), [`decode_scaled_fp8_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L375), [`decode_fp8_words`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L391), [`load_fp8_words`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L408), [`load_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L429), [`store_fp8_words`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L434), [`_reduce_tp_partial_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L449), [`emit_tp_reduce_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L506), [`emit_tp_all_gather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py#L657).

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py) · **Recent**

Source documentation: Configuration and workspace layouts for gfx950 A8W4 comm-fused MoE.

Symbols: [`Shape`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L16), [`_align_up`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L63), [`_mxfp8_scale_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L67), [`MegakernelConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L74), [`WindowConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L408), [`DirectConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py#L672).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py) · **Recent**

Source documentation: Direct BF16 reduce-scatter and shared-expert addition.

Symbols: [`_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py#L24), [`compile_reduce_scatter_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py#L42).

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py) · **Recent**

Source documentation: Single-launch Stage2 producer/consumer kernel. Producer workgroups run the natural-grid route GEMM. Completion is tracked per N tile; the final service_groups producers become communication consumers and execute the selected direct, reduce-broadcast, or reduce-scatter/all-gather path. This keeps the native GEMM tiling and avoids an artificial window.

Symbols: [`_decode_scaled_fp8_bf16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L42), [`_mxfp8_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L60), [`_store_mxfp8_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L77), [`_load_mxfp8_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L115), [`emit_service_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L150), [`compile_megakernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py#L1044).

Parameters referenced in this file: `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py) · **Recent**

Source documentation: Prepare route storage and adapt its ABI to the MXMoE GEMM2 producer.

Symbols: [`compile_mark_written_routes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L19), [`compile_clear_missing_routes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L52), [`resolve_route_input_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L92), [`_input_row_resolver_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L103), [`compile_megakernel_producer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L110), [`compile_window_producer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py#L138).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py) · **Recent**

Source documentation: Windowed GEMM2 and TP communication family.

Symbols: [`_producer_spatial_tag`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L34), [`_payload_tags`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L48), [`_producer_tile_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L58), [`_compose_compute`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L83), [`_compile_compute`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L184), [`compile_compute`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L193), [`_emit_local`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L199), [`_compose_cycle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L330), [`compile_cycle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L620), [`compile_drain`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py#L633).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py) · **Recent**

Source documentation: Compact dispatch path for MegaMoE v2 stage1.

Symbols: [`DispatchSlot`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L17), [`_load_fanout_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L77), [`_wave_inclusive_scan_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L110), [`_wave_reduce_max_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L126), [`_increment_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L138), [`_classify_fanout_wave_route`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L143), [`_configure_payload_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L241), [`_store_expert_metadata`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L285), [`_initialize_section_ready`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L316), [`_copy_token_row`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L367), [`_publish_tile_range`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L393), [`emit_direct_fixed_slot_payload`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L446), [`emit_direct_fixed_slot_finalize`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L570), [`_derive_allgather_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L668), [`_derive_next_fanout_pairs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L803), [`emit_dispatch_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L964), [`emit_dispatch_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L1401), [`emit_dispatch_payload`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L1552).

Parameters referenced in this file: `BLOCK_HIST`.

### `aiter/ops/flydsl/kernels/mega_moe/envs.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/envs.py) · **Recent**

Source documentation: Every environment variable MegaMoEV2 reads, in one place. Read through this module (envs.AITER_MEGA_...) rather than os.environ. A value is read when it is accessed, so the caller decides when it takes effect: at import, when a kernel is compiled, or when an instance is built.

Symbols: [`_flag`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/envs.py#L15), [`__getattr__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/envs.py#L41), [`__dir__`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/envs.py#L48).

### `aiter/ops/flydsl/kernels/mega_moe/gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py) · **Recent**

Source documentation: GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface.

Symbols: [`_LdsF32View`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py#L28), [`do_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py#L35), [`build_fused_gemm1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py#L287), [`compile_gemm1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py#L364), [`gemm1_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py#L476).

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mega_moe/gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py) · **Recent**

Source documentation: GEMM2 compute for fused MegaMoE v2 stage2.

Symbols: [`scale_view`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L31), [`scale_mma_atoms`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L49), [`mma_one_j`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L63), [`issue_a_load_lds_dt`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L101), [`gemm2_compute_v2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L158), [`_spart_output_tile_index`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L692), [`_resolve_g2_knobs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py#L731).

### `aiter/ops/flydsl/kernels/mega_moe/gemm_util.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py) · **Recent**

Source documentation: A8W4/A4W4 GEMM utilities for MegaMoE v2 Stage1.

Symbols: [`_load_row_map_subgroup`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L18), [`wait_lds_barrier`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L34), [`TileScheduler`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L40), [`ATileLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L59), [`AS2RLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L263), [`BWeightLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L303), [`BScaleLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L350), [`AScaleLoader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L372), [`MfmaScaleGU`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L496), [`_group_absmax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L641), [`_e8m0_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L651), [`SiluQuantEpilogue`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py#L660).

Parameters referenced in this file: `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py) · **Recent**

Source documentation: Static MegaMoEV2 configuration rules for MI355X.

Symbols: [`fixed_stage1_epoch_slot`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L70), [`fixed_stage1_epoch_slot_count`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L79), [`Stage1Config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L86), [`stage1_bundle_identity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L105), [`Stage2Config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L111), [`MegaMoEConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L133), [`Stage2BundleKey`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L154), [`MegaMoEBundleEntry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L171), [`MegaMoEBundlePlan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L179), [`nearest_token_bucket`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L201), [`mtpr_config_class`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L213), [`_fixed_dispatch_cu`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L217), [`_select_fixed_stage1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L229), [`_select_bounded_stage1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L247), [`_select_large_stage1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L285), [`_select_bounded_stage2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L331), [`_select_large_stage2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L372), [`_select_bucket_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L397), [`_replace_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L421), [`_TuningContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L435), [`_fp4_common_prelude`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L447), [`_tune_r1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L470), [`_tune_v4_pro`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L592), [`_tune_kimi_k3`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L733), [`NetworkProfile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L773), [`_match_network`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L810), [`select_mega_moe_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L831), [`build_mega_moe_bundle_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L973).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py) · **Recent**

Source documentation: Low-resource compact-dispatch prepare kernel for MegaMoE Stage1.

Symbols: [`compile_mega_moe_prepare`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L21), [`run_mega_moe_prepare`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L279), [`preload_mega_moe_prepare`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L306).

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py) · **Recent**

Source documentation: Fused Stage1 dispatch with FP8/FP4 activation and FP4-weight GEMM1.

Symbols: [`_Stage1KernelSpec`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L42), [`_validate_fixed_slot_geometry`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L52), [`_validate_dispatch_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L62), [`compile_mega_moe_stage1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L88), [`compile_mega_moe_stage1_bundle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L736), [`run_mega_moe_stage1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L874), [`run_mega_moe_stage1_bundle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L905), [`preload_mega_moe_stage1_bundle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L957).

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py) · **Recent**

Source documentation: Fused GEMM2 and weighted cross-rank P2P scatter.

Symbols: [`_fp8_scale_for_leader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L37), [`p2p_scatter_epilog`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L53), [`_stage2_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L289), [`compile_mega_moe_stage2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L298), [`_get_g2_launch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L641), [`run_mega_moe_stage2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L652), [`preload_mega_moe_stage2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L687).

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py) · **Recent**

Source documentation: Aligned common-row fusion for MegaMoE Stage2. The Stage1 fanout planner lays the selected experts' common routes out in the same order. This kernel consumes those two contiguous sections, evaluates the two experts sequentially with one accumulator set, and scatters their weighted sum once. The ordinary Stage2 kernel concurrently handles all residual rows.

Symbols: [`_store_accumulator_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py#L42), [`_pair_routewise_half_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py#L61), [`compile_mega_moe_stage2_aligned_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py#L220), [`run_mega_moe_stage2_aligned_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py#L560), [`preload_mega_moe_stage2_aligned_pair`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py#L594).

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py) · **Recent**

Source documentation: MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation.

Symbols: [`MegaMoEV2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31).

Parameters referenced in this file: `BLOCK_HIST`, `tile_k`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py) · **Recent**

Source documentation: FlyDSL 1x32 MXFP4/MXFP8 quantization with E8M0 scales.

Symbols: [`_load_i32x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L23), [`_store_i32x4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L29), [`emit_per_1x32_mx_fp8_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L35), [`emit_per_1x32_mx_fp4_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L75), [`build_per_1x32_mx_quant_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L114), [`_get_launcher`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L156), [`per_1x32_mx_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L165), [`build_mxfp4_moe_scale_sort_module`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L195), [`mxfp4_moe_scale_sort`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py#L276).

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py) · **Recent**

Source documentation: Barrier-wait and top-k reduce kernel for gfx1250 Stage2-fused MegaMoE.

Symbols: [`_make_combine_fused_sync`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py#L63), [`combine_reduce_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py#L132), [`_make_combine_fused_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py#L148).

Parameters referenced in this file: `num_warps`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py) · **Recent**

Source documentation: Send-side compact route plan for gfx1250 TDM dispatch. A low-LDS persistent-enough grid counts every local route into (dest_rank, local_expert) buckets, allgathers the histogram, then writes the destination compact row into tok_map. Dispatch copies payload straight onto that row, so the receiver never runs moe_route_g2l_lds.

Symbols: [`compact_plan_waves`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L42), [`_align32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L61), [`compact_hist_layout`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L65), [`compact_hist_stride`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L80), [`compact_done_nbytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L87), [`_wave32_inclusive_scan_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L93), [`compact_row_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L106), [`compile_tdm_compact_plan`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py#L121).

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/config.py) · **Recent**

Source documentation: gfx1250-only launch geometry for Stage2-fused MegaMoE.

Symbols: [`_select_dispatch_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/config.py#L59).

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py) · **Recent**

Source documentation: FlyDSL intranode EP dispatch driven by the gfx1250 Tensor Data Mover. A TDM moves a whole payload row on one descriptor. Routes are one lane each (WAVE / topk tokens per wave); a wave dedups a token's same-peer routes with a ballot + mbcnt_lo match-any. Recv slots are reserved one remote atomic per (block, peer) off an LDS histogram.

Symbols: [`_align`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L67), [`tdm_tokens_per_wave`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L71), [`tdm_stage_capacity`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L82), [`tdm_lds_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L93), [`_tile_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L99), [`tdm_max_warps`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L109), [`_make_dispatch_tdm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py#L130).

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py) · **Recent**

Source documentation: gfx1250 Stage2-fused MegaMoE host pipeline.

Symbols: [`_combine_tile`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L56), [`read_dispatch_wire_env`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L121), [`_DispatchWire`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L127), [`_align_up`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L154), [`_compact_gemm_align_m`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L158), [`_mori_dispatch_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L228), [`SymmetricArena`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L251), [`MegaMoEConfig`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L292), [`Routing`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L538), [`MegaMoEGfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py#L550).

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py) · **Recent**

Source documentation: Vendored TDM gather-mode API for gfx1250 (self-contained, no FlyDSL patch). Lets aiter kernels issue indexed TDM stores against a *stock* FlyDSL wheel -- i.e. without the FlyDSL-side change that taught make_tensor_gather_descriptor to accept a fly lds_view / pointer-based LDS index. Adapted from FlyDSL's expr/rocdl/tdm_ops.py (MI400 ISA S4.10.3.2 gather mode); it bottoms out on the low-level rocdl.tensor_store_from_lds 5-group intrinsic, which the wheel exposes.

Symbols: [`compute_padding_encoding`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L27), [`TDMGatherDescriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L51), [`_byte_offset_to_i64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L64), [`_zero_dgroup_v8i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L82), [`make_tensor_gather_descriptor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L87), [`make_tensor_gather_dgroup0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L240), [`tensor_store_gather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py#L286).

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py) · **Recent**

Source documentation: gfx1250 TDM (Tensor Data Mover) descriptors over RAW 64-bit addresses. tdm_ops.make_tensor_descriptor_2d derives the global base from a fly memref/tensor, which the EP dispatch cannot supply: a payload's destination is a *peer's* copy of a symmetric region, i.e. the integer lsa_ptr(pe, off), a runtime value rather than a kernel tensor argument. So GROUP0 is packed by hand from an i64 address plus an LDS byte address, and GROUP1 from the compile-time tile geometry.

Symbols: [`_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L66), [`tdm_group1`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L74), [`tdm_group1_rows_4b`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L107), [`tdm_plan_xfer_4b`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L134), [`tdm_group0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L170), [`tdm_load`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L196), [`tdm_store`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L201), [`tdm_wait`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py#L206).

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py) · **Recent**

Source documentation: Private host-side types for the gfx1250 MegaMoE pipeline.

Symbols: [`GpuPointerView`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py#L35), [`_from_gpu_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py#L46), [`Stage2ScatterContext`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py#L62).

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py) · **Recent**

Source documentation: Host side of the big-M path of MegaMoeTP.forward (the kernel: compile_bigm_kernel in mega_moe_tp_kernel.py): MegaMoeTPBigM (the SDMA queues and hipIpc arena, buffers, per-batch plans, launches), the disk cache of the compiled kernels, the bf16 row gather of the fused tail.

Symbols: [`_kernel_sources`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L61), [`_bigm_digest`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L93), [`_bigm_cache_dir`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L112), [`_warn_once`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L123), [`_private`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L129), [`_bigm_cached`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L140), [`_warn_spills`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L213), [`_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L229), [`_hip_check`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L257), [`_kcopy`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L262), [`_CudaBuf`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L270), [`MegaMoeTPBigM`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py#L282).

### `aiter/ops/flydsl/kernels/mega_moe_tp/common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py) · **Recent**

Source documentation: Device helpers shared by the MegaMoE TP and SpRsNorm kernels.

Symbols: [`_attr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L35), [`_u`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L39), [`i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L43), [`_LazyTy`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L50), [`_ty`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L60), [`lds_ptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L66), [`lds_ld`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L75), [`lds_ld_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L80), [`lds_st`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L84), [`lds_p`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L89), [`gptr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L96), [`lds_atomic_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L102), [`lds_atomic_or`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L110), [`lds_ld_acq`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L114), [`lds_st_rel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L126), [`lds_cas`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L130), [`_ctpop`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L136), [`g_ld_rel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L140), [`g_ld_sys`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L152), [`g_st_sys`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L156), [`g_add_agent`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L160), [`g_or_agent`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L164), [`g_ld_i32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L168), [`uni`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L172), [`uni64`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L176), [`rsrc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L185), [`bld`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L193), [`bst`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L197), [`_dma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L205), [`dma16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L220), [`dma4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L224), [`wait_vm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L228), [`wait_lgkm0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L232), [`amax`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L236), [`fp4_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L243), [`fp8x4_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L252), [`fp8x4_unpack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L263), [`e8_scale`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L273), [`fp8x8_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L277), [`mxfp8x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L283), [`scales4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L291), [`bf16_bits`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L299), [`pack_bf16x2`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L308), [`bf16x8_to_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L312), [`pack_bf16x8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L322), [`sum_live`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L328), [`wave_red`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L332), [`wave_rank`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L338), [`_swap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L347), [`swap16`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L356), [`swap32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L360), [`widen`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L364), [`mfma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L370), [`cat8`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L380), [`now`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L385), [`before`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L391), [`alive`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L397), [`spin_lds_ge`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L402), [`spin0`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L410), [`poll_sys_ge`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L417), [`i8x4_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L426), [`i8x4_unpack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L436), [`f32_bits_add`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L444), [`f32_bits_max`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L449), [`i4x8_pack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L453), [`i4x8_unpack`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L462), [`_read`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L472), [`tuned_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L479), [`nearest`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L491), [`device_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py#L503).

### `aiter/ops/flydsl/kernels/mega_moe_tp/communication.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/communication.py) · **Recent**

Source documentation: Per-instance device helpers (ctrl / peer addressing, LL packets, barriers, error reports, route gather) and the communication: the input all-gather, and the ReduceScatter of the GEMM2 partials (ar: plus the all-gather of the sums) with its final sums, LL packets, comm wave roles and the per-launch init / finish.

Symbols: [`build_communication`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/communication.py#L123).

### `aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py) · **Recent**

Source documentation: GEMM1 (gate / up + activation + quant of the intermediate into LDS) with its A loader wave, and GEMM2 (down projection) of a unit into the route regions.

Symbols: [`build_gemm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py#L74), [`bigm_lds`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py#L760), [`build_bigm_unit`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py#L786).

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py) · **Recent**

Source documentation: Host side of the single-kernel TP MegaMoE (launch configs, buffers, launches).

Symbols: [`LaunchCfg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L64), [`_allocation_base`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L89), [`_ipc_open`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L104), [`_ipc_close`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L112), [`_no_expandable_segments`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L121), [`_Slice`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L137), [`SymmetricArena`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L145), [`MegaMoeTPLDSError`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L249), [`_mt_table`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L257), [`MegaMoeTPEngine`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py#L263).

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py) · **Recent**

Source documentation: Compile-time layout (flag / ctrl / LDS slots), shape limits and the per-instance KernelCtx shared by the parts of the fused TP MegaMoE kernel.

Symbols: [`mega_moe_tp_shape_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L93), [`_nsk2_for`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L105), [`gemm2_group_step`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L111), [`gemm2_chunk_groups`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L117), [`mega_moe_tp_consts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L122), [`KernelCtx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L222), [`kernel_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L233), [`i8_row_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L414), [`unit_cap`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py#L418).

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py) · **Recent**

Source documentation: Fused TP MegaMoE kernel (a4w4, gfx950): the fused tail norm, the wave roles, the kernel entry and its launcher. The device code is built per instance by the build_* parts (communication, gemm, schedule) over one shared KernelCtx.

Symbols: [`compile_mega_moe_tp`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py#L114), [`compile_bigm_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py#L579).

### `aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py) · **Recent**

Source documentation: Unit scheduling: dynamic (active-expert row-chunk plan) and LB (two-phase GEMM1 / GEMM2 units claimed by whichever CTA is free).

Symbols: [`build_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py#L85).

### `aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py) · **Recent**

Source documentation: Sequence-parallel int8 reduce-scatter + residual add + RMSNorm (+ MoE router), one kernel.

Symbols: [`compile_sp_rs_norm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py#L58), [`SpRsNorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py#L633).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_comm_fused_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_comm_fused_moe.py)
- [op_tests/multigpu_tests/test_mega_moe_tp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_mega_moe_tp.py)
- [op_tests/tuners/tune_comm_fused_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuners/tune_comm_fused_moe.py)

## Performance and validation

Record TP/EP topology, expert placement, token counts and collective completion. A local GEMM speedup alone does not establish a fused-MoE gain.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
