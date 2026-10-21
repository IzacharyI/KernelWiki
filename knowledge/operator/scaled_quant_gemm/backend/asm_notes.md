---
title: "FP4 GEMM: packed inputs, fused output quantization and output reuse"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: asm
provider: aiter
source_commit: b3ad6c461703110dc36286265c16e453ef0e2dd2
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# FP4 GEMM input and output contracts

The [AITER FP4 GEMM wrapper](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/gemm_op_a4w4.py) at this snapshot exposes `gemm_mxfp4_asm` and `gemm_nvfp4_asm`. These ASM routes are guarded for gfx1250; do not transfer their layout or dispatch assumptions to gfx942/gfx950 paths. [Other backend references](../README.md) describe the wider mixed-precision GEMM family.

## Packed inputs are part of GEMM

MXFP4 uses two FP4 values per byte, so activation and weight storage have shapes `[M,K/2]` and `[N,K/2]`. Scales use E8M0 and 32-element groups along K. NVFP4 instead uses E4M3 scale inputs plus global scale factors. The selected preshuffle mode and the native dispatch determine the physical layout; matching only the payload dtype is insufficient.

Weight conversion can be performed once at load time. Activation conversion normally belongs to each invocation. Report both GEMM-only and complete projection latency when dynamic quantization or reshuffling is required.

## Fused output quantization

BF16 output is a plain `[M,N]` tensor. The MXFP8-output variant returns FP8 payload plus E8M0 scales, with one scale per 128 output columns. Scale storage pads M to a multiple of 64 and packs the logical scale matrix as `(Mpad/64, ceil(N/128), 16, 4)`. Use `unpack_mxfp8_out_scale` when comparing to a row-major reference; a direct reshape does not undo that permutation.

Fusing output quantization is useful only if the next consumer accepts the resulting payload and scale layout. Include any conversion between GEMMs in a pipeline comparison.

## Output reuse and empty batches

At this revision, keyword argument `y` lets the caller reuse a contiguous `[M,N]` output with matching dtype and device. MXFP8 scales remain a separately allocated output. For `M == 0`, the wrapper returns empty output buffers without launching the GEMM, after the architecture guard.

The [native ASM bridge](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/py_itfs_cu/asm_f4gemm.cu) and [FP4 tests](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_f4gemm.py) cover dispatch and validation. Check output aliasing, empty batches, preshuffle modes, scales and numerical accuracy with the matching GPU. These are source observations; KernelWiki has not measured a speedup or certified a best implementation.
