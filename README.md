# KernelWiki

An **operator-centered GPU kernel knowledge base** that documents backend implementations, applicability, performance evidence and optimization experience to help developers and GEAK write, select and improve kernels.

## Content

| Content | Start here | Purpose |
| --- | --- | --- |
| Written knowledge | [Operators](knowledge/operator/README.md) · [General](knowledge/general/README.md) | Implementation details, source entry points, performance and applicability, tuning methods and lessons from previous runs |
| Kernel implementations | [Source collection](kernels/README.md) · [Put, search and fetch API](kernelwiki/README.md) | Reuse complete source bundles with entry points, dependencies, correctness checks and benchmarks |

```text
knowledge/
  operator/<operator>/
    README.md
    backend/<backend>_aiter.md
    backend/<backend>_notes.md
    learned_knowledge/
  general/
    profile_bottleneck/
    optimization/
    backend/
    hardware/
    quantization/
    ...
kernels/<operator>/<backend>/<implementation>/
  kernel.py
  entry.py
  test.py
  benchmark.py
  metadata.yaml
  README.md
kernelwiki/
  client.py
```

## Getting started

- [Dense GEMM](knowledge/operator/dense_gemm/README.md) · [Quantized GEMM](knowledge/operator/scaled_quant_gemm/README.md) · [Grouped GEMM / MoE](knowledge/operator/grouped_gemm_moe/README.md)
- [MLA](knowledge/operator/mla_attention/README.md) · [Paged decode attention](knowledge/operator/attention_decode_paged/README.md) · [RMSNorm](knowledge/operator/rmsnorm/README.md)
- [Bottleneck analysis](knowledge/general/profile_bottleneck/README.md) · [Optimization methods](knowledge/general/optimization/README.md) · [Writing knowledge and recording evidence](knowledge/README.md)

## Initial content

- GEAK source: [`02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/tree/02bfbb0b72746941be3ab3ad48fb21a284ae2fde). Imports 894 Markdown documents, including 166 kernel and end-to-end experience cards, together with their supporting recipe resources.
- AITER source: [`e9995854abbd`](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca). Adds 159 operator/backend implementation references covering 919 implementation and support source files, of which 824 changed in the preceding three months. Another 3,271 ASM/HSA artifacts and metadata files are referenced by family.
- There are 63 operator directories. Source-reference coverage and measured-performance coverage are separate: this import adds no GPU benchmark results and does not certify historical SOTA labels as current winners.
- Previously processed GEAK inbox files, old generated indexes and index-generation scripts are excluded from the curated knowledge. Their originals remain available in the pinned GEAK revision.

AITER is the main source of implementation knowledge. Documents include pinned links to source, configuration and tests, with the focus on the kernels themselves. Choosing the best implementation requires a specific GPU, dtype, shape/layout and comparable measurements.

## Kernel source collection

The repository now stores complete kernel implementations alongside their explanatory knowledge. The first source bundle is [AITER's row-tiled Triton RMSNorm](kernels/rmsnorm/triton/aiter_row_tiled/README.md), with its kernel body, standalone entry point, launch configurations, numerical reference, correctness checks and benchmark program. It retains AITER's MIT license and pinned source attribution.

This initial bundle is a source reference with upstream-reported performance. It has not been remeasured on the packaging host and is not labeled `best_known`. A best-known entry must identify exact measured cases and attach correctness/performance evidence to its source digest. See the [source-bundle contract](kernels/README.md).

## Using the source API

```bash
python -m pip install -e .
```

```python
from kernelwiki import KernelClient

client = KernelClient.local("kernels")
result = client.search(
    "rmsnorm", backend="triton", gfx="gfx950", dtype="bf16",
    shape={"M": 16384, "N": 128},
)
item = result["implementations"][0]
client.fetch(
    item["implementation_id"], revision=item["revision"],
    destination="downloaded_rmsnorm",
)
```

For remote candidates, set `GEAK_KB_URL` and use `KernelClient.from_env()`. The existing `GEAK_KB_STORE_URL`, `KB_STORE_URL` and token variables remain supported. `put()` uploads a complete source bundle; `fetch()` returns the same runnable directory layout. New source records use `geak:kernel_code:...` identities. The previous patch-specific Python API has been removed in version 0.2. Historical remote patch records are not modified or presented as complete source implementations.

Repository entries are curated with Git. The remote service can retain multiple candidate revisions and measurement artifacts. Both use the same implementation ID, source digest and bundle revision. Read text knowledge directly from Markdown. See the [API guide](kernelwiki/README.md) for publication, search and retrieval details.

## Development checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Attribution and licenses

Imported GEAK content retains its attribution and Apache-2.0 license. AITER references and excerpts retain their MIT license notice. See [LICENSE](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
