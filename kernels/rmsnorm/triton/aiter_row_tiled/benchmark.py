# SPDX-License-Identifier: MIT
"""Write one exact-case measurement; never update best-known status automatically."""

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import statistics

import torch
import triton
import triton.testing

from kernelwiki import SourceBundle
from entry import rmsnorm
from reference import rmsnorm_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--baseline", choices=["torch", "aiter_generic"], default="torch")
    parser.add_argument("--output", type=Path, default=Path("measurement.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("A ROCm GPU is required; no performance result was produced.")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(1729)
    x = torch.randn((args.m, args.n), device="cuda", dtype=dtype)
    weight = torch.randn(args.n, device=x.device, dtype=dtype)
    eps = 1e-6
    candidate = lambda: rmsnorm(x, weight, eps)
    baseline = lambda: rmsnorm_reference(x, weight, eps)
    baseline_files = {"reference.py": Path(__file__).with_name("reference.py")}
    if args.baseline == "aiter_generic":
        from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_forward
        baseline = lambda: _rmsnorm_forward(x, weight, eps)[0]
        kernel_module = importlib.import_module("aiter.ops.triton._triton_kernels.normalization.rmsnorm")
        baseline_files = {"wrapper.py": Path(inspect.getfile(_rmsnorm_forward)),
                          "kernel.py": Path(kernel_module.__file__)}
    expected = rmsnorm_reference(x, weight, eps)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    for call in (candidate, baseline):
        torch.testing.assert_close(call(), expected, rtol=tolerance, atol=tolerance)
    baseline_times, candidate_times = [], []
    for iteration in range(6):
        arms = [(baseline, baseline_times), (candidate, candidate_times)]
        if iteration % 2:
            arms.reverse()
        for call, values in arms:
            values.append(1000 * triton.testing.do_bench(call, warmup=100, rep=300, return_mode="median"))
    device = torch.cuda.get_device_properties(x.device)
    environment = {"gpu_model": device.name, "rocm": torch.version.hip,
                   "torch": torch.__version__, "triton": triton.__version__}
    protocol = {"method": "triton.testing.do_bench", "warmup_ms": 100, "rep_ms": 300,
                "statistic": "median of six alternating-arm medians", "eps": eps,
                "baseline": args.baseline,
                "baseline_source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for name, path in baseline_files.items()},
                "environment": environment, "shape": {"M": args.m, "N": args.n}, "dtype": args.dtype}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    bundle = SourceBundle.load(Path(__file__).parent)
    result = {"kind": "measured", "gfx": device.gcnArchName.split(":", 1)[0],
              "dtype": args.dtype, "shape": protocol["shape"], "bench_key": key,
              "baseline": args.baseline, "latency_us": statistics.median(candidate_times),
              "baseline_latency_us": statistics.median(baseline_times), "correctness": "passed",
              "environment": environment, "source_digest": bundle.source_digest,
              "evidence": [str(args.output)], "protocol": protocol,
              "baseline_samples_us": baseline_times, "candidate_samples_us": candidate_times}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
