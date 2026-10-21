# sparse_attention_nsa: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [MiniMax-M3 / NSA block-sparse GQA attention → the live path IS an editable in-tree Triton kernel](e2e-sparse-attn-nsa-triton-gfx950.md)
- [Block-sparse GQA attention → the concrete Triton→Triton rewrite that lands the win (both archs)](e2e-sparse-attn-triton2triton-opt-gfx942-gfx950.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
