---
origin: geak
evidence_scope: end_to_end
operator: dense_gemm
key: dense bf16 GEMM · gfx942 · sglang
type: lever
confidence: ★★★
effect: +2.23% e2e VERIFIED (1548.9→1583.5 tok/s, non-overlapping 5-repeat A/B); ~+6% cumulative w/ attn-triton
confirms: 2
last_seen: 2026-06-08
---
# Dense bf16 GEMM → tune aiter's per-shape DB (the #1 verified e2e win on this stack)
- lever: the live dense-GEMM path on sglang/gfx942 is aiter `tuned_gemm.py` (executing hipBLASLt
  `Cijk_*`), seam `aiter.tuned_gemm:gemm_a16w16`. Tune its per-shape DB — this is THE GEMM lever, and
  the strongest *transferred-to-e2e* win recorded.
- apply: capture real shapes `AITER_TUNE_GEMM=1` → `gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>`
  → deploy `AITER_CONFIG_GEMM_BF16=<tuned.csv>` (pure env, no package edit). FlyDSL races inside this DB
  (`libtype=flydsl`) and is auto-selected where it wins.
- verify: `AITER_LOG_TUNED_CONFIG=1` → count `is tuned on cu_num` hits (>0 = engaged; the winning run
  had 246 hits). The capture's correct `bias=False` + full shape coverage is what makes it both ENGAGE
  and WIN — a bias-mismatched/partial tune reads ~0/−0.6% (superseded).
- caution: NOT TunableOp / `HIPBLASLT_TUNING_FILE` — aiter bypasses the PyTorch/hipBLASLt C dispatch
  for its tuned shapes, so those hooks don't touch the live path.
- caution (STACK-SPECIFIC, verify the seam): this lever assumes the live GEMM goes through
  `aiter.tuned_gemm:gemm_a16w16`. On **vLLM** (≥0.19, gfx942) the dense bf16 path is
  `vllm...layers.utils:rocm_unquantized_gemm_impl`, which for these Qwen3-14B (N,K) families routes to
  `torch.nn.functional.linear` → hipBLASLt and **does not call aiter.tuned_gemm** (`use_aiter_triton_gemm()`
  returns False for all 4 families on gfx942; on_gfx950 paths off). So `AITER_CONFIG_GEMM_BF16` would get
  **0 engagement on vLLM** — confirm the seam before tuning. There, Tier-B yields nothing (TunableOp ties
  box-default hipBLASLt isolated); the lever is the **Tier-C author** route rebound at
  `rocm_unquantized_gemm_impl` (existing editable `aiter.ops.triton.gemm_a16w16` is a good baseline, but
  gated OFF for these shapes). [gfx942 · vLLM dense bf16, 2026-06-22 Qwen3-14B]
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-08 (verified A/B, full recipe in `SKILL_DIR/knowledge/gemm_tuning/aiter_gemm_tuning.md`);
  vLLM-seam caution: exp/e2e_Qwen-Qwen3-14B_20260622 bake-off (seam-inspected, 0-engagement predicted)

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow/knowledge/learned/aiter-bf16-tuned-gemm-gfx942.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
