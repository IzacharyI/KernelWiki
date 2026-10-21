# AITER hip: shared implementation support

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

Source snapshot: `e9995854abbd935f0190b19abebe9938b6b90cca`. These files provide shared infrastructure, bindings or unclassified support; their presence is not counted as a documented operator implementation.

- [csrc/include/aiter_ctypes_error.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_ctypes_error.h)
- [csrc/include/aiter_dispatch.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_dispatch.h)
- [csrc/include/aiter_enum.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_enum.h)
- [csrc/include/aiter_hip_common.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_hip_common.h)
- [csrc/include/aiter_logger.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_logger.h)
- [csrc/include/aiter_operator.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_operator.h)
- [csrc/include/aiter_opus_plus.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_opus_plus.h)
- [csrc/include/aiter_stream.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_stream.h)
- [csrc/include/aiter_tensor.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/aiter_tensor.h)
- [csrc/include/custom.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/custom.h)
- [csrc/include/dispatch_utils.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/dispatch_utils.h)
- [csrc/include/hip_float8.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/hip_float8.h)
- [csrc/include/hip_float8_impl.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/hip_float8_impl.h)
- [csrc/include/ps.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/ps.h)
- [csrc/include/rocm_ops.hpp](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/rocm_ops.hpp)
- [csrc/include/warp_sort.h](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/warp_sort.h)

## Sources

- [AITER](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
