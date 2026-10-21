# gqa_mqa_attention: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [fp8 KV storage with bf16 in-register dequant on scatter-bound paged attention](kernel-fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound.md)
- [Pack heads per workgroup, then strip the inner loop](kernel-pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [Retile to one program per query, then delete every per-trip rendezvous](kernel-retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
- [Six axes that stayed closed on a graph-replay-timed sparse prefill attention op](kernel-six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)
- [Traffic is the only live axis once attention is scatter-bound on HBM](kernel-traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
