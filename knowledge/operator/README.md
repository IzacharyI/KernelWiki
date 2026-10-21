# Operator knowledge

Browse an operator, then its backend implementation and learned knowledge. AITER is the main source for current implementations; GEAK contributes prior implementation explanations, optimization methods and historical experience.

| Operator | AITER backend/provider pages | Learned cards |
| --- | --- | ---: |
| [act_and_mul_silu_gelu](act_and_mul_silu_gelu/README.md) | aiter, flydsl, gluon, hip, triton | 0 |
| [alibi](alibi/README.md) | GEAK reference only | 0 |
| [all_to_all_dispatch_combine](all_to_all_dispatch_combine/README.md) | aiter, hip | 0 |
| [allgather](allgather/README.md) | triton | 0 |
| [allreduce](allreduce/README.md) | aiter, asm, flydsl, hip | 0 |
| [argmax_topk](argmax_topk/README.md) | aiter, asm, flydsl, hip, triton | 0 |
| [attention_decode_paged](attention_decode_paged/README.md) | asm, flydsl, gluon, hip, triton | 35 |
| [attention_indexer](attention_indexer/README.md) | aiter, flydsl, gluon, hip, opus, triton | 0 |
| [attention_prefill_fmha](attention_prefill_fmha/README.md) | aiter, asm, flydsl, gluon, hip, triton | 1 |
| [batched_gemm](batched_gemm/README.md) | aiter, ck, flydsl, gluon, triton | 0 |
| [cast_fill_copy](cast_fill_copy/README.md) | GEAK reference only | 4 |
| [causal_conv1d](causal_conv1d/README.md) | aiter, flydsl, hip, triton | 0 |
| [chunked_prefill](chunked_prefill/README.md) | GEAK reference only | 0 |
| [compressed_attention](compressed_attention/README.md) | flydsl | 0 |
| [context_parallel_attention](context_parallel_attention/README.md) | GEAK reference only | 0 |
| [conv2d](conv2d/README.md) | ck, flydsl, triton | 0 |
| [conv3d](conv3d/README.md) | flydsl, triton | 0 |
| [cross_entropy](cross_entropy/README.md) | triton | 0 |
| [cumsum_scan](cumsum_scan/README.md) | GEAK reference only | 0 |
| [dense_gemm](dense_gemm/README.md) | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 16 |
| [depthwise_conv](depthwise_conv/README.md) | GEAK reference only | 0 |
| [elementwise](elementwise/README.md) | hip, triton | 0 |
| [embedding](embedding/README.md) | GEAK reference only | 0 |
| [fused_add_rmsnorm](fused_add_rmsnorm/README.md) | gluon, triton | 0 |
| [fused_allreduce_rmsnorm](fused_allreduce_rmsnorm/README.md) | GEAK reference only | 0 |
| [fused_moe_grouped_gemm](fused_moe_grouped_gemm/README.md) | aiter, flydsl | 0 |
| [fused_norm_quant](fused_norm_quant/README.md) | aiter, hip | 3 |
| [gather_scatter](gather_scatter/README.md) | GEAK reference only | 0 |
| [gemm_epilogue_fused](gemm_epilogue_fused/README.md) | triton | 0 |
| [gqa_mqa_attention](gqa_mqa_attention/README.md) | GEAK reference only | 5 |
| [grouped_gemm_moe](grouped_gemm_moe/README.md) | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 37 |
| [groupnorm](groupnorm/README.md) | aiter, hip | 0 |
| [hstu_attention](hstu_attention/README.md) | flydsl, triton | 0 |
| [kv_cache_quant](kv_cache_quant/README.md) | GEAK reference only | 0 |
| [layernorm](layernorm/README.md) | aiter, hip, triton | 0 |
| [layout_shuffle](layout_shuffle/README.md) | aiter, flydsl, triton | 0 |
| [linear_attention_gated_delta](linear_attention_gated_delta/README.md) | aiter, flydsl, gluon, hip, triton | 6 |
| [lm_head_logits](lm_head_logits/README.md) | GEAK reference only | 0 |
| [mhc](mhc/README.md) | aiter, flydsl, gluon, hip, triton | 0 |
| [mla_attention](mla_attention/README.md) | aiter, asm, flydsl, gluon, hip, opus, triton | 0 |
| [moe_dispatch_combine](moe_dispatch_combine/README.md) | flydsl | 0 |
| [moe_routing_topk](moe_routing_topk/README.md) | aiter, flydsl, hip, opus, triton | 4 |
| [mrope](mrope/README.md) | GEAK reference only | 0 |
| [paged_kv_copy](paged_kv_copy/README.md) | aiter, gluon, hip, triton | 0 |
| [qk_norm_rope_cache](qk_norm_rope_cache/README.md) | aiter, flydsl, gluon, hip, triton | 0 |
| [quant_dequant_fp8](quant_dequant_fp8/README.md) | aiter, asm, flydsl, gluon, hip, triton | 8 |
| [quant_fp4_mxfp](quant_fp4_mxfp/README.md) | aiter, flydsl, gluon, hip, triton | 0 |
| [quant_int8](quant_int8/README.md) | GEAK reference only | 0 |
| [reduce_scatter](reduce_scatter/README.md) | triton | 0 |
| [reduction](reduction/README.md) | asm, hip | 0 |
| [rmsnorm](rmsnorm/README.md) | aiter, hip, opus, triton | 0 |
| [rope](rope/README.md) | aiter, hip, triton | 0 |
| [sampling_topk_topp](sampling_topk_topp/README.md) | aiter, hip | 0 |
| [scaled_quant_gemm](scaled_quant_gemm/README.md) | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 13 |
| [shared_expert_fusion](shared_expert_fusion/README.md) | GEAK reference only | 0 |
| [skinny_gemv_decode](skinny_gemv_decode/README.md) | GEAK reference only | 0 |
| [sliding_window_attention](sliding_window_attention/README.md) | GEAK reference only | 0 |
| [softmax](softmax/README.md) | triton | 0 |
| [sparse_attention_nsa](sparse_attention_nsa/README.md) | aiter, gluon, hip, opus, triton | 2 |
| [speculative_decode_verify](speculative_decode_verify/README.md) | GEAK reference only | 0 |
| [splitk_streamk_gemm](splitk_streamk_gemm/README.md) | asm, flydsl, triton | 0 |
| [symmetric_tridiagonalization](symmetric_tridiagonalization/README.md) | hip | 0 |
| [transpose](transpose/README.md) | flydsl, triton | 0 |
