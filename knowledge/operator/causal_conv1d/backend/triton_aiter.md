---
title: "Causal one-dimensional convolution: AITER triton"
kind: implementation_knowledge
operator: causal_conv1d
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Causal one-dimensional convolution — AITER triton

Prefill and token-update paths differ in state movement, padding and Q/K/V splitting. The update path mutates a rolling history that must remain correct over subsequent tokens.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/conv/causal_conv1d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d.py) · **Recent**

Source documentation: support varlen + continuous batching when x is 2D tensor x: (dim,cu_seq_len) cu_seq_len = total tokens of all seqs in that batch sequences are concatenated from left to right for varlen weight: (dim, width) conv_states: (...,dim,width - 1) itype updated inplace if provided [it use cache_indices to get the index to the cache of conv_state for that sequence conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True and after that …

Symbols: [`_flydsl_conv1d_update_enabled`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d.py#L16), [`_assert_implemented_width`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d.py#L25), [`causal_conv1d_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d.py#L42), [`causal_conv1d_update`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d.py#L220).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`.

### `aiter/ops/triton/gated_delta_net/causal_conv1d_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_decode.py) · **Recent**

Source documentation: Triton decode causal_conv1d_update (fused split q/k/v) — public host entry. Decode-stage counterpart of the prefill causal_conv1d_prefill module: a single-/few-token autoregressive conv update that directly outputs split q/k/v. The @triton.jit / Gluon kernels live in aiter.ops.triton._triton_kernels.gated_delta_net.decode.causal_conv1d_update_split_qkv.

Symbols: [`causal_conv1d_update_split_qkv`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_decode.py#L25).

Parameters referenced in this file: `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py) · **Recent**

Source documentation: Triton prefill causal-conv1d (fused split q/k/v) — public host entry points. This is the public Triton namespace for the Gated-Delta-Rule (GDR / Mamba) prefill front-end depthwise conv. It exposes two interchangeable Triton kernels; the upper-layer framework selects the backend explicitly: * causal_conv1d_split_qkv_triton_fn -- 1D per-token kernel (portable; any width).

Symbols: [`causal_conv1d_split_qkv_triton_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py#L43), [`_build_chunk_schedule`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py#L139), [`causal_conv1d_split_qkv_triton_tile_fn`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py#L165).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py) · **Recent**

Symbols: [`_causal_conv1d_fwd_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py#L27), [`_causal_conv1d_update_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py#L407).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py) · **Recent**

Symbols: [`_ba_source_offsets`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py#L16), [`_z_source_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py#L33), [`_feat_source_idx`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py#L52), [`_causal_conv1d_update_single_token_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py#L94), [`_reshape_causal_conv1d_update_single_token_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/conv/causal_conv1d_update_single_token.py#L307).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_Z`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py) · **Recent**

Source documentation: Decode causal_conv1d_update (fused split q/k/v) — Triton/Gluon kernels. Host-side launch wrapper lives in the public namespace aiter.ops.triton.gated_delta_net.causal_conv1d_decode.

Symbols: [`_causal_conv1d_update_split_qkv_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L16), [`load_conv_weights`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L254), [`load_conv_states`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L272), [`tuple_combine`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L290), [`gluon_causal_conv1d_update_split_qkv_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L295), [`gluon_causal_conv1d_update_split_qkv_kernel_notuple`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/causal_conv1d_update_split_qkv.py#L475).

Parameters referenced in this file: `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/causal_conv1d_fwd_split_qkv.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/causal_conv1d_fwd_split_qkv.py) · **Recent**

Source documentation: Prefill causal conv1d (fused split q/k/v) — Triton @triton.jit kernels. Two interchangeable kernels: a 1D per-token kernel (serial token loop, any conv width) and a 2D-tiled kernel (vectorized over feature/token, conv width in {2, 3, 4}). Host-side launch wrappers live in the public namespace aiter.ops.triton.gated_delta_net.causal_conv1d_prefill.

Symbols: [`_causal_conv1d_fwd_split_qkv_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/causal_conv1d_fwd_split_qkv.py#L16), [`_causal_conv1d_fwd_split_qkv_tile_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/causal_conv1d_fwd_split_qkv.py#L317).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/conv/causal_conv1d_update_single_token.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d_update_single_token.py)

Source documentation: Launchers for **causal conv1d update** single-token paths. causal_conv1d_update_single_token updates conv_state **in place** inside the Triton kernel (the "update" in the name), then writes the convolution output into x/out as in vLLM.

Symbols: [`_default_conv_state_indices`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d_update_single_token.py#L24), [`causal_conv1d_update_single_token`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d_update_single_token.py#L28), [`fused_reshape_causal_conv1d_update_single_token`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/conv/causal_conv1d_update_single_token.py#L153).

Parameters referenced in this file: `BLOCK_N`, `BLOCK_Z`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/triton_tests/conv/test_causal_conv1d.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_causal_conv1d.py)
- [op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py)
- [op_tests/triton_tests/utils/causal_conv1d_update_refs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/causal_conv1d_update_refs.py)
- [op_tests/test_causal_conv1d_prefill_split_qkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_prefill_split_qkv.py)
- [op_tests/test_causal_conv1d_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_update.py)
- [op_tests/test_flydsl_causal_conv1d_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_causal_conv1d_update.py)

## Performance and validation

Test a sequence against the full convolution reference. Include state initialization, optional activation and variable sequence boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
