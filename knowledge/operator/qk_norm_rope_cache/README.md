# qk_norm_rope_cache

5 current AITER implementation references; 0 historical experience cards.

## AITER implementations

Source snapshot: `e9995854abbd`. These pages describe source and validation routes; they do not imply that every implementation has been benchmarked on every GPU.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Fused Q/K normalization, RoPE and cache writes — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Fused Q/K normalization, RoPE and cache writes — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Fused Q/K normalization, RoPE and cache writes — AITER gluon](backend/gluon_aiter.md) |
| hip | [Fused Q/K normalization, RoPE and cache writes — AITER hip](backend/hip_aiter.md) |
| triton | [Fused Q/K normalization, RoPE and cache writes — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

## Validation

Validate Q and K outputs plus modified cache locations. Include position tails, partial rotary dimensions and the quantization scale layout.
