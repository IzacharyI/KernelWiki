---
title: "AITER MLA: cache layout, causal queries and backend dispatch"
kind: implementation_notes
operator: mla_attention
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_review
---

# AITER MLA: cache layout, causal queries and backend dispatch

`mla_decode_fwd` selects among multiple execution paths using architecture, dtype, head count, page size, cache layout and scheduling inputs. The provider-level entry name alone does not identify the launched backend.

## Byte layout and indexing are separate choices

The wrapper distinguishes `token_major` from `kv_shuffled` cache byte layouts. At this snapshot, shuffled gfx1250 paths support FP8 Q/KV and BF16 output with physical page size 64. QH64 uses CSR metadata, a 768-byte Q-head stride, multiple queries and split-K reduction; QH128 uses dense sequence lengths/page tables and one query without splitting.

Within the shuffled QH64 path, `kv_csr_layout="page"` indexes physical pages; `"token"` uses slot ids. Token CSR does not change the cache's byte layout or physical page size. The contract requires sequences to start at offset zero and contain consecutive slots within a logical page; arbitrary sparse token gathers are not supported by that path.

## Multi-query causality

For shuffled QH64, query i sees `max(0, kv_len - q_len + i + 1)` tokens under bottom-right causal masking. Fully masked queries produce zero output and negative-infinity LSE. Testing a single query cannot expose all masking errors in multi-token decode.

Round-robin context parallelism introduces another position mapping: local token j maps to global position `j * cp_world_size + cp_rank`. Check the actual path's CP support; shuffled kernels explicitly exclude CP and persistent scheduling in their documented contract.

## Backend selection and timing

Some gfx1250 page-size-one FP8 head-count cases can use shipped code objects exported from FlyDSL, controlled by `AITER_MLA_DECODE_PS1_ASM`; other cases use FlyDSL JIT, OPUS or other paths. Thus “ASM execution” does not necessarily mean hand-written assembly provenance.

Warm up compilation and cached CPU-scale transfers before graph capture or supply GPU scales. Benchmark the whole selected path, including metadata preparation and split reduction where applicable. Record the dispatch conditions and validate QH64/QH128 return contracts separately.

No new performance measurement is attached. These contracts identify cases that must remain correct when replacing or tuning the implementation.

## Sources

- [MLA public wrapper and dispatch](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/mla.py)
- [Current implementation and test references](aiter_aiter.md)
