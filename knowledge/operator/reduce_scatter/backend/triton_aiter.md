---
title: "Reduce scatter: AITER triton"
kind: implementation_knowledge
operator: reduce_scatter
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Reduce scatter — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py) · **Recent**

Source documentation: Fused Reduce-Scatter + RMSNorm + Quantization + All-Gather This module implements a fully fused Triton kernel that performs: 1. Reduce-scatter: Pull-based via iris.load 2. RMSNorm: Root mean square normalization on local shard 3. Quantization: Optional per-token quantization (FP8) 4. All-gather: Push-based via iris.put Based on Iris example: examples/22_rs_rmsnorm_fp8quant_ag/reduce_scatter_rmsnorm_quant_fused.py

Symbols: [`_quantize_fp8_stage`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py#L47), [`fused_pipeline_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py#L84), [`reduce_scatter_rmsnorm_quant_all_gather`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py#L228).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/comms/reduce_scatter.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/reduce_scatter.py) · **Recent**

Source documentation: Reduce-Scatter communication primitive using Iris. This module provides a reduce-scatter operation along the M dimension using GPU-initiated communication via the Iris library.

Symbols: [`_reduce_scatter_impl`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/reduce_scatter.py#L30), [`_reduce_scatter_kernel`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/reduce_scatter.py#L120), [`reduce_scatter`](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/comms/reduce_scatter.py#L165).

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_reduce_scatter.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_reduce_scatter.py)
- [op_tests/multigpu_tests/triton_test/test_reduce_scatter_all_gather.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/triton_test/test_reduce_scatter_all_gather.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
