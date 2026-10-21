# AITER row-tiled RMSNorm

This bundle contains the actual Triton kernel and a standalone forward wrapper. It computes `x * rsqrt(mean(x**2, dim=-1) + eps) * weight`, accumulates in FP32 and returns the input dtype.

## Applicability

- GPU: gfx942 or gfx950 with a ROCm PyTorch/Triton environment.
- Inputs: contiguous `x[M,N]` and `weight[N]`, matching BF16 or FP16 dtype and device.
- Shapes: `M > 8192`, `1 <= N <= 2048`.
- Forward inference only, without gradients or in-place input updates.

The row-tiled grid exposes `ceil(M / BLOCK_M)` independent programs. `BLOCK_N` is the next power of two of N and `BLOCK_M = max(8, min(16384 / BLOCK_N, 32))`. The two shipped architecture configurations both use eight warps and two stages, copied from the pinned AITER source.

## Run

Use ROCm-compatible PyTorch and its matching Triton build. Install KernelWiki from the repository if you want the benchmark to record source digests. In this bundle directory:

```bash
python test.py
python benchmark.py --m 16384 --n 128 --dtype bf16 --baseline torch --output measurement.json
```

```python
import torch
from entry import rmsnorm

x = torch.randn((16384, 128), device="cuda", dtype=torch.bfloat16)
weight = torch.ones(128, device=x.device, dtype=x.dtype)
y = rmsnorm(x, weight, eps=1e-6)
```

The Torch baseline is a mathematical reference, not AITER's best implementation. With an installed AITER, `--baseline aiter_generic` measures its generic RMSNorm path explicitly. It does not silently substitute another baseline. The benchmark records the baseline source-file hash, environment, shape, timing protocol and kernel source digest.

To publish a measurement, add its result to `metadata.yaml` and declare the local result file in `files.support`, or use a durable external evidence reference. Record a passed correctness run separately before changing the bundle's validation status. Evidence is not added or promoted automatically by the benchmark program.

## Evidence

The original AITER wrapper reports a 13.6x forward ratio at M=16384, N=128, BF16 on MI308X. This is an upstream source comment, not a measurement of this extracted bundle. There are no new GPU timings or passed-GPU-test claims here. The bundle remains `status: reference`; `best_known` requires measured, case-specific evidence tied to its source digest.

`test.py` checks non-power-of-two widths, tail rows, zero input, two epsilon values, both dtypes and unsupported-shape rejection. The packaging host had no ROCm GPU, so these tests are supplied for execution on the target machine.

## Source and adaptations

- [Original kernel](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py)
- [Original wrapper and performance comment](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/normalization/rmsnorm.py)
- [Original naming helper](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/utils/_triton/kernel_repr.py)
- [gfx942 launch config](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json)
- [gfx950 launch config](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/normalization/rmsnorm_large_m_small_n/DEFAULT.json)

Kernel arithmetic and the naming helper are copied unchanged. Imports, the standalone wrapper, validation programs and packaging are adapted for KernelWiki. [LICENSE](LICENSE) preserves AITER's MIT license. The wrapper accepts a narrower input contract than AITER's general dispatcher and performs no hidden contiguous copies.
