---
title: "Cross entropy: AITER triton"
kind: implementation_knowledge
operator: cross_entropy
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Cross entropy — AITER triton

Vocabulary-wide reductions combine log-sum-exp with label selection, and may produce gradients. Ignore indices, reduction mode and numerical stability are observable semantics.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/cross_entropy.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/cross_entropy.py) · **Recent**

Source documentation: Vocab-parallel cross-entropy: Python API wrapping Triton kernels. This module provides cross_entropy_forward and cross_entropy_backward that orchestrate kernel launches and all_gather communication so that higher-level frameworks (e.g. Lumen, Megatron) only need a single function call.

Symbols: [`_block_size_and_warps`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/cross_entropy.py#L34), [`cross_entropy_forward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/cross_entropy.py#L41), [`cross_entropy_forward_chunked`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/cross_entropy.py#L179), [`cross_entropy_backward`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/cross_entropy.py#L336).

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/cross_entropy.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/cross_entropy.py) · **Recent**

Source documentation: Triton JIT kernels for vocab-parallel cross-entropy loss. Three kernels are provided: 1. _ce_local_softmax_stats_kernel — per-rank online softmax to compute local max, denominator, and X_y (the logit at the target index). 2. _ce_fused_loss_grad_kernel — merges per-rank softmax stats (after all_gather), computes the loss, and writes the gradient into the logits tensor in-place. 3.

Symbols: [`_ce_local_softmax_stats_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/cross_entropy.py#L37), [`_ce_fused_loss_grad_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/cross_entropy.py#L89), [`_ce_grad_scale_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/cross_entropy.py#L200).

Parameters referenced in this file: `BLOCK_SIZE`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_cross_entropy.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_cross_entropy.py)
- [op_tests/triton_tests/test_cross_entropy.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_cross_entropy.py)
- [op_tests/multigpu_tests/triton_test/test_cross_entropy_tp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/triton_test/test_cross_entropy_tp.py)

## Performance and validation

Test extreme logits, ignored labels and backward results with the same reduction mode. Report vocabulary size and row count.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
