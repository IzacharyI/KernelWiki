---
title: Benchmarks & agents — pointer doc (e2e_workflow, GEAK, examples, KernelBench/TritonBench)
kind: reference
updated: 2026-06-08
sources:
  - e2e_workflow/README.md
  - https://github.com/AMD-AGI/GEAK
  - perf_knowledge/index/sota_registry.yaml
---

# Benchmarks & agents — pointer doc

Where the *agents* and *benchmark harnesses* around perf_knowledge live, and how they consume this knowledge
base. perf_knowledge is the **reference layer**; the items below are the **actors** that read it. This file is a
map, not a tutorial — each target owns its own docs.

## e2e_workflow (the e2e optimizer workflow)
Path (repo-relative): [`e2e_workflow/`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow)
(entry `e2e_workflow.js`, `README.md`, `roles/`, `knowledge/`, `scripts/`).

A deterministic JS-orchestrated multi-agent **Workflow** that raises sglang/vLLM serving throughput on
MI300X. It is a *system layer* on top of the unchanged single-kernel `kernel_workflow`; it profiles a
running server, does Amdahl triage (`pct_gpu_time × achievable_speedup`), tunes config/backend knobs
first, extracts hot kernels into standalone unittests, optimizes them via the kernel layer, overlays
them back, and re-validates e2e throughput.

Roles (in `roles/`) — these are the perf_knowledge consumers:
- **system_architect** — strategy/routing; reads the SOTA landscape to decide config-vs-kernel tracks.
- **op_benchmarker** — *named in this task* — standardizes per-op benchmark shapes/metrics. (On-box the
  comparable role files are `op_benchmarker.md` plus `profiler.md`; **(verify exact role name)** against
  the installed `roles/` if scripts depend on it.)
- **config_tuner** — Tier-0 env/flag/backend sweep (uses [`env_vars.md`](env_vars.md)).
- **profiler**, **kernel_extractor**, **e2e_integrator**, **director** — trace, extract, reintegrate, gate.

## How the architect / op_benchmarker query perf_knowledge
The machine-queryable surface is [`../index/sota_registry.yaml`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/index/sota_registry.yaml) —
an auto-generated mirror of `sota_matrix.md` (one entry per operator×backend with
`status / gens / dtypes / regimes / card / sources`). Intended query pattern:

> *"best backend for operator X on gen Y, dtype Z, regime R"* → filter `entries[]` by those keys,
> take `status: sota`, follow `card:` to the SOTA card.

- **system_architect** filters the registry by the served model's hot ops (from the profile Top-N) to
  decide whether a known-SOTA path already exists (config track) or a kernel needs authoring (kernel
  track), and reads the env knobs in [`env_vars.md`](env_vars.md) for the Tier-0 sweep.
- **op_benchmarker** resolves an op id to its SOTA card's *how-to-bench* section for canonical shapes,
  dtypes, and the perf-number format ([`../index/conventions.md`](geak_index/conventions.md)).

Always edit card frontmatter and regenerate the registry (`index/_gen_registry.py`) — never hand-edit
`sota_registry.yaml`.

## GEAK (AMD agentic kernel generator)
In-repo: the repository root (`examples/`, `mcp_tools/`, `docs/`).
Upstream: https://github.com/AMD-AGI/GEAK.

LLM agent that generates + iteratively refines Triton kernels for AMD GPUs, with a debugging/reflection
loop and an evaluator. perf_knowledge feeds GEAK target SOTA characteristics and pitfalls; GEAK is invoked by
the kernel layer when a hot op has no SOTA path. Authoring workflow:
[`../workflows/authoring_a_kernel_with_geak.md`](../workflows/authoring_a_kernel_with_geak.md).

## KernelBench / TritonBench (external eval harnesses)
- **KernelBench** — https://github.com/ScalingIntelligence/KernelBench — task suite for LLM-generated
  GPU kernels (correctness + speedup vs. reference). Used as an external yardstick for GEAK-style
  generation, not part of the serving loop.
- **TritonBench** — https://github.com/pytorch-labs/tritonbench — Triton operator microbenchmark
  suite; the per-op shapes/metrics op_benchmarker mirrors should stay consistent with TritonBench
  conventions where an op overlaps.

Both are *external* benchmarks; perf_knowledge's authoritative numbers come from on-box runs in `examples/`
recorded in the [`../index/conventions.md`](geak_index/conventions.md) perf format.

## Sources
- e2e_workflow README + roles — [`e2e_workflow/README.md`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow/README.md), `roles/`.
- GEAK — repo root ; https://github.com/AMD-AGI/GEAK
- registry/query contract — [`../index/sota_registry.yaml`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/index/sota_registry.yaml), [`../index/conventions.md`](geak_index/conventions.md)
- KernelBench — https://github.com/ScalingIntelligence/KernelBench ; TritonBench — https://github.com/pytorch-labs/tritonbench
- Items marked **(verify exact name)** were not confirmed verbatim at write time.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/reference/benchmarks_and_agents.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
