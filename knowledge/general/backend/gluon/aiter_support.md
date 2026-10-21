# AITER gluon: shared implementation support

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

Source snapshot: `e9995854abbd935f0190b19abebe9938b6b90cca`. These files provide shared infrastructure, bindings or unclassified support; their presence is not counted as a documented operator implementation.

- [aiter/ops/triton/_gluon_kernels/common/utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/common/utils.py)

## Sources

- [AITER](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
