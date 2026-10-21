# cast_fill_copy: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Collapse the host launch path first on a dispatch-bound scatter, then re-grid](kernel-collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both.md)
- [Four host-side axes that a dispatch-bound tiny op has already paid for](kernel-four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
- [GPU-side knobs are a closed axis once submit dominates](kernel-gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound.md)
- [Raw driver launch for a dispatch-bound copy op](kernel-raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
