---
title: "Argmax topk: AITER aiter"
kind: implementation_knowledge
operator: argmax_topk
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Argmax topk — AITER aiter

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py) · **Recent**

Source documentation: Unified fused topk gating for MoE routing. Args: score_func: one of {"sqrtsoftplus" (DeepSeek V4-Pro default), "sigmoid" (Llama4), "softmax" (DeepSeek V3 / classic MoE)}. This scores the routed prefix only. correction_bias: optional bias tensor, pass None for no bias. Must be float32, or bfloat16 when gating_output is not float16. Sized to the routed expert count, not the shared columns.

Symbols: [`topk_gating_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L20), [`_can_use_gfx1250_exact_sigmoid_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L37), [`_gfx1250_exact_sigmoid_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L84), [`_use_flydsl_herd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L117), [`_valid_bias_dtypes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L146), [`topk_gating`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L156), [`biased_grouped_topk_hip`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L232), [`grouped_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L245), [`gen_moe_fused_gate_fake_tensor`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L257), [`_moe_fused_gate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L278), [`moe_fused_gate`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L291), [`biased_grouped_topk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L318), [`biased_grouped_topk_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L400), [`grouped_topk_torch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L445), [`_top_k_per_row_prefill`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L485), [`topk_mb_workspace_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L501), [`topk_ob_workspace_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L507), [`topk_use_mulblocks`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L513), [`_get_topk_mb_workspace_keyed`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L517), [`get_topk_mb_workspace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L524), [`get_topk_scratch_workspace`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L548), [`top_k_per_row_prefill`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L567), [`top_k_per_row_prefill_fast`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L643), [`_top_k_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L656), [`_flydsl_topk_decode_shape_supported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L701), [`_should_use_flydsl_topk_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L718), [`_hip_top_k_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L769), [`top_k_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L798), [`_top_k_per_row_decode_packed`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L926), [`flydsl_dcp_topk_merge`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L968), [`flydsl_radix_topk_one_block_prefill`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L1005), [`flydsl_radix_topk_one_block_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L1036), [`flydsl_top_k_per_row_decode`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L1067), [`top_k_per_row_decode_fast`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk.py#L1103).

### `aiter/ops/topk_plain.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_plain.py) · **Recent**

Source documentation: Would a rowStarts/rowEnds call stay on the batched launcher? Mirrors should_use_topk_radix in csrc/kernels/topk_plain_kernels.cu, which the variable-length AdaptiveTopK consults. When it holds, one batched launch serves every row.

Symbols: [`_topk_plain`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_plain.py#L17), [`topk_plain_workspace_size`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_plain.py#L32), [`topk_plain_batches_ragged_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_plain.py#L39), [`topk_plain`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_plain.py#L60).

### `aiter/ops/topk_select.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py) · **Recent**

Source documentation: Per-row top-k with a DeepSelect-shaped interface, over four backends. aiter carries four per-row selectors, each fastest in a different corner and none able to cover the whole domain: argmax k=1 only, and a reduction rather than a selection: the row split across as many workgroups as it takes to fill the part. Owns k=1, and is the only one to take bf16/fp16 as well as fp32. small_k one chunk per lane, so k <= wave_size, and a survivor buffer that grows with the row bound.

Symbols: [`_unsupported_arch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L153), [`_full_rows`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L161), [`_no_range`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L172), [`_gather_selected_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L178), [`_gather_selected`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L207), [`_available`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L233), [`_choose`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L294), [`_plain_takes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L323), [`_decode_takes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L329), [`topk_select_backend`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L338), [`_reject_unsupported`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L393), [`topk_select`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L457), [`_stream_split_parts`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L667), [`_stream_scratch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L688), [`_dispatch`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/topk_select.py#L699).

Parameters referenced in this file: `BLOCK_K`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_flydsl_topk_per_row_small_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_topk_per_row_small_k.py)
- [op_tests/test_topk_per_row.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row.py)
- [op_tests/test_topk_per_row_stable.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row_stable.py)
- [op_tests/test_topk_plain.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_plain.py)
- [op_tests/test_topk_row_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_row_prefill.py)
- [op_tests/test_topk_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_select.py)
- [op_tests/op_benchmarks/triton/bench_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_topk.py)
- [op_tests/triton_tests/test_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_topk.py)
- [op_tests/triton_tests/torch_compile/test_compile_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_topk.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
