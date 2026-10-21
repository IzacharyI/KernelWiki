# KernelWiki

An **operator-centered GPU kernel knowledge base** that documents backend implementations, applicability, performance evidence and optimization experience to help developers and GEAK write, select and improve kernels.

## Content

| Content | Start here | Purpose |
| --- | --- | --- |
| Written knowledge | [Operators](knowledge/operator/README.md) · [General](knowledge/general/README.md) | Implementation details, source entry points, performance and applicability, tuning methods and lessons from previous runs |
| Kernel patches | [Write and search API](kernel_patch/README.md) | Store, discover and download patches and tuning artifacts through the existing GEAK KB service |

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
kernel_patch/
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

## Using the patch API

```bash
python -m pip install -e .
export GEAK_KB_URL="https://your-existing-kb-service"
# Use your existing GEAK_KB_STORE_TOKEN / KB_STORE_TOKEN configuration.
```

```python
from kernel_patch import PatchClient, kernel_addresses

client = PatchClient.from_env()
addresses = kernel_addresses(
    kernel_name="_rms_norm_kernel", backend="triton", gfx="gfx950", rocm_version="7.3"
)
result = client.search(canonical_ids=addresses, top_k=5)
```

The API supports `GEAK_KB_STORE_URL`, `KB_STORE_URL` and existing canonical IDs. See [kernel_patch](kernel_patch/README.md) for write and download examples, search scope and error handling. Read written knowledge directly from this repository's Markdown files; the Python package provides patch access.

## Development checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Attribution and licenses

Imported GEAK content retains its attribution and Apache-2.0 license. AITER references and excerpts retain their MIT license notice. See [LICENSE](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
