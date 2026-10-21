# SPDX-License-Identifier: MIT
"""Run correctness and boundary checks on a supported ROCm GPU."""

import torch

from entry import rmsnorm
from reference import rmsnorm_reference


def main():
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("A ROCm GPU is required; no correctness result was produced.")
    torch.manual_seed(1729)
    cases = [(8193, 1), (8193, 127), (16384, 128), (16384, 512),
             (16384, 1024), (16384, 1280), (8193, 2048)]
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        for m, n in cases:
            for zero in (False, True):
                x = torch.zeros((m, n), device="cuda", dtype=dtype) if zero else torch.randn((m, n), device="cuda", dtype=dtype)
                weight = torch.randn(n, device=x.device, dtype=dtype)
                for eps in (1e-6, 1e-3):
                    actual = rmsnorm(x, weight, eps)
                    expected = rmsnorm_reference(x, weight, eps)
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    count += 1
    for m, n in ((8192, 128), (8193, 2049)):
        x = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(n, device=x.device, dtype=x.dtype)
        try:
            rmsnorm(x, weight)
        except ValueError:
            pass
        else:
            raise AssertionError("an unsupported shape was accepted")
    torch.cuda.synchronize()
    print(f"Passed {count} correctness cases and unsupported-shape checks.")


if __name__ == "__main__":
    main()
