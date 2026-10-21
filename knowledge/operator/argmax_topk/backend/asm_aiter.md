---
title: "Argmax topk: AITER asm"
kind: implementation_knowledge
operator: argmax_topk
backend: asm
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Argmax topk — AITER asm

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

AITER ships architecture-specific code objects and dispatch metadata. Binary family names identify candidates, but do not prove supported shapes or performance; follow the wrapper and metadata that select an object.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

## Assembly code-object families

These are shipped code objects and their metadata, grouped by family rather than treating each compiled tile as a new operator. This records provenance and selection material; assembly internals have not been reconstructed from binaries.

| Architecture / family | Files | Changed in the period |
| --- | ---: | ---: |
| [hsa/gfx942/topk_per_row_decode](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/hsa/gfx942/topk_per_row_decode) | 1 | 0 |
| [hsa/gfx942/topk_per_row_prefill](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/hsa/gfx942/topk_per_row_prefill) | 1 | 0 |
| [hsa/gfx942/topksoftmax](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/hsa/gfx942/topksoftmax) | 25 | 3 |
| [hsa/gfx950/topksoftmax](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/hsa/gfx950/topksoftmax) | 25 | 3 |

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_topk.py)
- [op_tests/test_flydsl_topk_per_row_small_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_topk_per_row_small_k.py)
- [op_tests/test_topk_per_row.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row.py)
- [op_tests/test_topk_per_row_stable.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_per_row_stable.py)
- [op_tests/test_topk_plain.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_plain.py)
- [op_tests/test_topk_row_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_row_prefill.py)
- [op_tests/test_topk_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_select.py)
- [op_tests/triton_tests/test_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_topk.py)
- [op_tests/triton_tests/torch_compile/test_compile_topk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_topk.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
