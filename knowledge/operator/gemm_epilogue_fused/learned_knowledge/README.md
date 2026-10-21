# gemm_epilogue_fused: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Axes that stayed closed on a roof-bound fused norm-GEMM path](kernel-axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [Collapse the dispatch chain inside each shape-regime arm](kernel-collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
- [Cut VALU on the prefill arm with native casts and packed dot](kernel-cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
