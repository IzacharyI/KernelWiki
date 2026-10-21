# dense_gemm: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Dense bf16 GEMM → tune aiter's per-shape DB (the #1 verified e2e win on this stack)](e2e-aiter-bf16-tuned-gemm-gfx942.md)
- [bf16 dense GEMM on sglang/gfx950 — backend swap dead-ends, the aiter DB tune engages](e2e-dense-gemm-bf16-gfx950-sglang.md)
- [gfx950 vLLM fp8 block-scale dense GEMM — the AUTHORED Triton lever, not just the tune DB](e2e-fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)
- [MXFP8 E8M0 DENSE linear GEMM (vLLM native, gfx950) — author/rewrite Triton, tune BLOCK_K/N](e2e-mxfp8-dense-linear-gemm-gfx950.md)
- [MXFP8 (E8M0 1×32 microscale) GEMM heads → editable Triton `tl.dot_scaled` tiles](e2e-mxfp8-microscale-gemm-gfx950.md)
- [A hand-written loop has to out-schedule, not out-structure, the compiler](kernel-a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
- [Bitcast the fp8 flavour the matrix pipe actually has](kernel-bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both.md)
- [Cache the per-call host work when the host owns a large share of a tiny op's wall](kernel-cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode.md)
- [Gluon register-staged wide-K MFMA for fp16 dense GEMM](kernel-gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [Graph capture of a single tiny dispatch can be a higher floor than the dispatch it replaces](kernel-graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode.md)
- [Measure the launch floor before buying a device-side round on a tiny decode op](kernel-measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)
- [Occupancy axis closes when the backend pins waves-per-eu](kernel-occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)
- [Once it routes to tuned vendor assembly, out-generating it is a closed axis](kernel-once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)
- [Own the dispatch layer, then race backends behind it](kernel-own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound.md)
- [Size the exposed host residue before buying a launch-overhead round](kernel-size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound.md)
- [The in-source ceiling of an MFMA-bound dense GEMM](kernel-the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
