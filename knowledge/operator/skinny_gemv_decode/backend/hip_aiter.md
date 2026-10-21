---
title: "Small-M decode GEMM: three HIP dispatch paths"
kind: implementation_knowledge
operator: skinny_gemv_decode
backend: hip
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Small-M decode GEMM: three HIP dispatch paths

[AITER tuned GEMM](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/tuned_gemm.py) implements `skinny_gemm` for decode-sized projections. It selects three distinct native routes through [the custom-op bindings](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/custom.py):

| Solution index | Native entry | Wrapper behavior |
| --- | --- | --- |
| `0` | `wvSpltK` | Passes token count and CU count to the wave Split-K path. |
| `1` | `LLMM1` | Uses `rows_per_block=4`. |
| `2` | `wv_splitk_small_fp16_bf16` | Uses the specialized small FP16/BF16 route. |

The [native kernel bodies](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/custom_kernels.cu) contain small/medium/large wave Split-K specializations. These were previously hidden in the generic custom-kernel category.

## Selection and applicability

The default skinny-shape predicate requires FP16/BF16 and `K % 8 == 0`. Its regimes depend on CU count: M up to 4 permits K up to 9216, M from 5 to 8 permits K up to 5120, and M from 9 to 16 permits K up to 256, with different N limits. Read `is_skinny_default_shape` for the exact inequalities; these are dispatch heuristics, not a blanket speed guarantee or the complete callable domain.

`skinny_gemm` rejects pre-shuffled weights. It allocates output in the input dtype, then applies optional bias and output conversion separately. Include those operations and allocation policy when comparing to a GEMM with a fused epilogue. The presence of `wvSplitKQ` in the same binding file does not make it one of these three dispatch routes.

## Evidence and validation

- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_gemm_a16w16.py)

Inspected at AITER `b3ad6c461703`. The variants above describe source implementations and dispatch contracts, not independently measured winners. Shared headers and generated instances are implementation components, not additional operators. No GPU tests or benchmarks were run for this page. Record the selected implementation, GPU, source/toolchain revisions, shapes, layouts, timing scope and correctness evidence before comparing performance.
