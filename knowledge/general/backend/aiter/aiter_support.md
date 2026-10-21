# AITER aiter: shared implementation support

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

Source snapshot: `e9995854abbd935f0190b19abebe9938b6b90cca`. These files provide shared infrastructure, bindings or unclassified support; their presence is not counted as a documented operator implementation.

- [aiter/benchmark_data_init.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/benchmark_data_init.py)
- [aiter/benchmark_reporting.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/benchmark_reporting.py)
- [aiter/bert_padding.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/bert_padding.py)
- [aiter/int4_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/int4_utils.py)
- [aiter/ops/aiter_operator.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/aiter_operator.py)
- [aiter/ops/communication.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/communication.py)
- [aiter/ops/custom.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/custom.py)
- [aiter/ops/enum.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/enum.py)
- [aiter/ops/gradlib.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gradlib.py)
- [aiter/ops/prefill_batch_metadata.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/prefill_batch_metadata.py)
- [aiter/paged_attn.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/paged_attn.py)
- [aiter/smi_monitor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/smi_monitor.py)
- [aiter/test_common.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/test_common.py)

## Sources

- [AITER](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
