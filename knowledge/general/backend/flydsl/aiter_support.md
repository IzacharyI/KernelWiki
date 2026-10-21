# AITER flydsl: shared implementation support

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

Source snapshot: `e9995854abbd935f0190b19abebe9938b6b90cca`. These files provide shared infrastructure, bindings or unclassified support; their presence is not counted as a documented operator implementation.

- [aiter/ops/flydsl/kernels/buffer_ops.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/buffer_ops.py)
- [aiter/ops/flydsl/kernels/communication_ops_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/communication_ops_utils.py)
- [aiter/ops/flydsl/kernels/dpp_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/dpp_utils.py)
- [aiter/ops/flydsl/kernels/gfx1250_cluster.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gfx1250_cluster.py)
- [aiter/ops/flydsl/kernels/kernels_common.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/kernels_common.py)
- [aiter/ops/flydsl/kernels/mfma_epilogues.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mfma_epilogues.py)
- [aiter/ops/flydsl/kernels/tdm_ops_gfx1250.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/tdm_ops_gfx1250.py)
- [aiter/ops/flydsl/kernels/tensor_shim.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/tensor_shim.py)
- [aiter/ops/flydsl/kernels/utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/utils.py)

## Sources

- [AITER](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
