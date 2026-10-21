# moe_routing_topk: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Bypass the JIT launcher for a dispatch-bound Triton op](kernel-bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
- [Dispatch-floored router select: spend the budget on the host lane first](kernel-dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
- [Graph capture loses to a direct launch when the graph holds one tiny kernel](kernel-graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
- [The device lane on a small router top-k is close to closed](kernel-the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
