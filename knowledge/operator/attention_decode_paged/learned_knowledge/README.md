# attention_decode_paged: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [sglang triton decode attention — the split-KV count is the op-level lever, and one flag is a trap](e2e-decode-attn-splitkv-triton-gfx950-sglang.md)
- [vLLM paged attention with a non-pow2 KV block → the live path is the editable in-tree Triton kernel](e2e-paged-attn-nonpow2-gfx942.md)
- [sglang + aiter paged decode: the seam is `aiter:paged_attention_ragged`, and it has NO op-level knob](e2e-paged-attn-ragged-aiter-gfx950-sglang.md)
- [The "non-editable CK paged-decode head has no op-level knob" prior is WRONG on sglang+aiter](e2e-paged-attn-ragged-qkvversion-gfx950-sglang.md)
- [A single-partition control separates a rejected KV split from a wrong KV split](kernel-a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)
- [Ask whether the traffic is removable before you tune how it is cached](kernel-ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)
- [Axes that close once decode attention sits on its read roof](kernel-axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [Buy prefetch depth with a global-to-LDS DMA on bandwidth-bound decode attention](kernel-buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [Choose the non-temporal hint per operand, not per kernel](kernel-choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [Collapse the co-resident sequence set to break the KV address phase](kernel-collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode.md)
- [Collapse the partition grid instead of optimizing the round trip it creates](kernel-collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [Decode attention: pay the host tax first, then halve KV bytes, then re-tune occupancy](kernel-decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
- [Decode attention: the Python/ctypes prologue is the first three rounds](kernel-decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode.md)
- [Derive the split-KV decode launch shape from a constant-byte tile and a WG/CU fill target](kernel-derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [Dispatch collapse first, then per-regime specialisation, on latency-bound decode attention](kernel-dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
- [Drop the non-temporal cache hint on once-read KV streams](kernel-drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
- [Enable the source's own dormant split-KV path before authoring a new one](kernel-enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)
- [Four axes that stayed closed on a latency-floored paged decode kernel](kernel-four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [Four host and compute directions that a latency-floored decode kernel gives back](kernel-four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
- [Geometry, occupancy and load width are a spent axis here](kernel-geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode.md)
- [Get the nt bit onto KV loads by loading one native 128-bit vector](kernel-get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
- [High LDS-wait counters next to a high roofline fraction can be a compute tail](kernel-high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [Instructions and registers are not currency at near-total memory stall](kernel-instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)
- [Launch-meta first on latency-floored paged decode, and let the backend pick occupancy](kernel-launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
- [Measure the empty-graph replay floor before funding a capture round](kernel-measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)
- [One dispatch for split-KV decode, and the protocol that pays for it](kernel-one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
- [Only adding or removing a dependency moves a tuned paged attention](kernel-only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode.md)
- [Positive cache counters and a cheaper launcher can both buy zero clock](kernel-positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
- [Price the host fraction before spending a round on the launch floor](kernel-price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [Price the residual before funding fusion or geometry work at the roof](kernel-price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
- [Reproduce the golden's own rounding before costing a KV reassociation](kernel-reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)
- [Split-KV decode: the two-dispatch shape is welded, budget the round elsewhere](kernel-split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode.md)
- [Split only up to one workgroup per CU, and make pipeline depth a function of launched WGs](kernel-split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
- [The residual launch axis on decode attention closes once host time is far under GPU time](kernel-the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
- [Unclamp the KV tile from the page size, then de-rate it by head width](kernel-unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
