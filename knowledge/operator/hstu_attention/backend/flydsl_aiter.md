---
title: "HSTU attention: AITER flydsl"
kind: implementation_knowledge
operator: hstu_attention
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# HSTU attention — AITER flydsl

HSTU attention has its own score transform and masking semantics, with separate forward and backward implementations. A standard softmax-attention reference is not interchangeable.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/hstu_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py) · **Recent**

Source documentation: High-level FlyDSL HSTU Attention Forward API.

Symbols: [`_str2bool`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L45), [`_problem_key`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L79), [`_tuned_config_map`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L106), [`_get_tuned_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L171), [`_get_default_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L203), [`_compile_launcher`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L253), [`_validate_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L322), [`flydsl_hstu_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L396), [`_validate_bwd_inputs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L464), [`_bwd_tuned_config_map`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L542), [`_get_bwd_tuned_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L611), [`_build_balance_perm`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L641), [`_get_bwd_default_config`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L691), [`_compile_bwd_launcher`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L745), [`flydsl_hstu_attention_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L822), [`_make_bwd_kernel_runners`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L938), [`FlydslHstuAttention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L1060), [`flydsl_hstu_attention`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/hstu_attention.py#L1134).

Parameters referenced in this file: `waves_per_eu`.

### `aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd.py) · **Recent**

Source documentation: hstu_attention_bwd - FlyDSL KV-owned **fused** backward (dV + dK in one pass) Backward of HSTU attention.

Symbols: [`validate_hstu_attention_bwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd.py#L67), [`build_hstu_attention_bwd_dvdk`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd.py#L169).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_THREADS`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd_dq.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd_dq.py) · **Recent**

Source documentation: hstu_attention_bwd_dq - FlyDSL kernel (causal-only; computes dQ) Companion to hstu_attention_bwd.py. That kernel is KV-owned and produces dV/dK (both reduce over the query index).

Symbols: [`build_hstu_attention_bwd_dq`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_bwd_dq.py#L64).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_THREADS`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py) · **Recent**

Source documentation: Shared helpers for the FlyDSL HSTU attention forward and backward kernels. Two kinds of helpers live here so there is a single source of truth across the forward, the KV-owned (dV/dK) backward, and the Q-owned (dQ) backward kernels: * Host-side geometry constants and small host helpers (dtype mapping, per-arch DMA/swizzle parameters, LDS capacity) shared by all HSTU kernel builders.

Symbols: [`_arch_mfma_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L50), [`_mfma_params_for_dim`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L68), [`_dtype_to_elem_type`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L76), [`_arch_dma_params`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L84), [`lds_cap_bytes`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L101), [`exp2_f32`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L107), [`make_mfma_acc`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L117), [`bind_mfma_accs`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L139), [`pack_mfma_frag`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L152), [`decode_lane`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L174), [`grouped_loader`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L196), [`swz_col`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L210), [`make_lds_dma`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py#L218).

### `aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py) · **Recent**

Source documentation: hstu_attention_fwd - FlyDSL kernel out_i = (1/N) * sum_j valid(i,j) * silu(alpha * q_i * k_j^T) * v_j (N = max_seq_len) HSTU forward (causal / non-causal, windowed, contextual, targets; any head_dim/hidden_dim % 16). GEMM2 consumes P as operand A and row-major V as operand B, so V stages without a transposed LDS scatter. K stages via LDS; V loads to registers under a counted vmcnt, then publishes to LDS for GEMM2.

Symbols: [`validate_hstu_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py#L61), [`build_hstu_attention_fwd`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py#L152).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_THREADS`, `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_hstu_attn.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_hstu_attn.py)
- [op_tests/test_flydsl_hstu_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hstu_attention.py)
- [op_tests/test_flydsl_hstu_attention_bwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hstu_attention_bwd.py)
- [op_tests/triton_tests/attention/test_hstu_attention_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_hstu_attention_ref.py)
- [op_tests/triton_tests/attention/test_hstu_attn.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_hstu_attn.py)
- [op_tests/triton_tests/utils/hstu_attention_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/hstu_attention_ref.py)

## Performance and validation

Use AITER HSTU tests for the selected mask and jagged sequence layout. Validate gradients separately from forward output.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
