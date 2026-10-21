# AITER opus: shared implementation support

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

Source snapshot: `e9995854abbd935f0190b19abebe9938b6b90cca`. These files provide shared infrastructure, bindings or unclassified support; their presence is not counted as a documented operator implementation.

- [aiter/ops/opus/_arch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/_arch.py)
- [aiter/ops/opus/dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/dispatch.py)
- [aiter/ops/opus/launch_plan.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/launch_plan.py)
- [aiter/ops/opus/policy.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/policy.py)
- [csrc/include/opus/dtypes.hpp](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/dtypes.hpp)
- [csrc/include/opus/hip_minimal.hpp](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/hip_minimal.hpp)
- [csrc/include/opus/opus.hpp](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/opus/opus.hpp)

## Sources

- [AITER](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
