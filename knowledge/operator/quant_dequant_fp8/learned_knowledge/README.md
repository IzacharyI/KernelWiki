# quant_dequant_fp8: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Axes that stay closed once a quant-cast graph sits at two nodes](kernel-axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [Collapse the graph nodes first, then shape-gate a single-workgroup fusion](kernel-collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [Delete the in-kernel bounds guard from the host before declaring a scheduling transform unprofitable](kernel-delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
- [Divide by the group scale is a correctly-rounded reciprocal, not a division](kernel-divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both.md)
- [Fuse the quant passes behind a tag-slot grid barrier](kernel-fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [Near the practical HBM ceiling the bandwidth knobs are a closed axis](kernel-near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)
- [Reinterpret a frozen launch through an exported wrapper object](kernel-reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [Software-emulated fp8 cast: find it by differential recompile, kill it with the native convert](kernel-software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
