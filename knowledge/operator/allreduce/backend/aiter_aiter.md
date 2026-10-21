---
title: "Allreduce: AITER aiter"
kind: implementation_knowledge
operator: allreduce
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Allreduce — AITER aiter

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/custom_all_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py) · **Recent**

Source documentation: Launch fused custom AllReduce + MHC post (no pre / RMSNorm).

Symbols: [`init_custom_ar`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L15), [`all_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L27), [`reduce_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L39), [`all_gather_reg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L53), [`all_gather_unreg`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L62), [`fused_allreduce_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L73), [`fused_allreduce_rmsnorm_pad`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L89), [`fused_allreduce_rmsnorm_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L105), [`fused_allreduce_rmsnorm_quant_per_group`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L123), [`fused_allreduce_rmsnorm_mxfp4_quant`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L143), [`fused_qknorm_allreduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L161), [`fused_qknorm_allreduce_rope`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L176), [`dispose`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L195), [`meta_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L199), [`register_input_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L203), [`register_output_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L209), [`get_graph_buffer_count`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L215), [`get_graph_buffer_ipc_meta`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L219), [`register_graph_buffers`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L223), [`allocate_meta_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L229), [`allocate_data_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L233), [`free_meta_buffer`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L237), [`get_meta_buffer_ipc_handle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L241), [`init_custom_ar_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L248), [`all_reduce_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L259), [`reduce_scatter_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L271), [`all_gather_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L285), [`p2p_bw_test_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L296), [`dispose_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L309), [`meta_size_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L313), [`register_input_buffer_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L317), [`register_output_buffer_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L323), [`init_custom_ar_gfx1250_ipc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L334), [`register_input_buffer_gfx1250_ipc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L346), [`register_output_buffer_gfx1250_ipc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L352), [`get_graph_buffer_count_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L358), [`get_graph_buffer_ptrs_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L362), [`register_graph_buffers_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L366), [`start_sync_latency_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L370), [`end_sync_latency_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L374), [`two_sync_latency_gfx1250`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L378), [`fused_allreduce_mhc_post_only`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L382), [`fused_allreduce_mhc_post_one_stage`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L397), [`fused_allreduce_mhc_post_split`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L412), [`_launch_fused_allreduce_mhc_post`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L426), [`launch_fused_allreduce_mhc_post_only`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L462), [`launch_fused_allreduce_mhc_post_split`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom_all_reduce.py#L489).

### `aiter/ops/quick_all_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py) · **Recent**

Symbols: [`init_custom_qr`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L14), [`qr_destroy`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L23), [`qr_uses_pull_q4`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L27), [`qr_all_reduce`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L31), [`qr_all_reduce_rmsnorm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L41), [`qr_get_handle`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L56), [`qr_open_handles`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L60), [`qr_max_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_all_reduce.py#L64).

### `aiter/ops/quick_reduce_scatter.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_reduce_scatter.py) · **Recent**

Source documentation: Single-phase INT4 reduce-scatter using QuickReduce IPC state.

Symbols: [`qr_reduce_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/quick_reduce_scatter.py#L11).

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/gfx1250_poc/test_gfx1250_allreduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/gfx1250_poc/test_gfx1250_allreduce.py)
- [op_tests/multigpu_tests/test_custom_allreduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce.py)
- [op_tests/multigpu_tests/test_custom_allreduce_capture_registration.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce_capture_registration.py)
- [op_tests/multigpu_tests/test_custom_allreduce_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_custom_allreduce_fp8.py)
- [op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py)
- [op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4_rmsnorm.py)
- [op_tests/multigpu_tests/test_quick_all_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce.py)
- [op_tests/multigpu_tests/test_quick_all_reduce_gfx1250.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce_gfx1250.py)
- [op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_quick_all_reduce_rmsnorm.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
