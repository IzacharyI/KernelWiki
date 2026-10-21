# Kernel patch: write, search and fetch

This package uses the existing GEAK KB service. Metadata stays in that service and files use its object storage. KernelWiki does not commit run artifacts or operate a new server.

## Configuration

```bash
python -m pip install -e .
export GEAK_KB_URL="https://your-existing-kb-service"
```

URL precedence is `GEAK_KB_URL`, `GEAK_KB_STORE_URL`, then `KB_STORE_URL`; blank values are ignored. Authentication continues to use `GEAK_KB_STORE_TOKEN`, falling back to `KB_STORE_TOKEN`. Use your existing credential provisioning. `GEAK_KB_READ_ONLY=1` allows reads and blocks writes.

## Addressing

```python
from kernel_patch import PatchClient, kernel_addresses

client = PatchClient.from_env()
addresses = kernel_addresses(
    kernel_name="_rms_norm_kernel",
    backend="triton",
    gfx="gfx950",
    rocm_version="7.3.1",
)
```

This preserves GEAK's two addresses:

```text
geak:kernel:gfx950:_rms_norm_kernel:triton:rocm:7.3
geak:kernel:gfx950:_rms_norm_kernel:triton:rocm
```

Use the canonical kernel symbol that the original producer used, not an extraction filename such as `triton_<symbol>_task.py`. For an existing record, its recorded `canonical_id` is authoritative and can be supplied directly. Operator names organize textual knowledge; they do not replace old kernel identities.

## Write

The producer supplies the actual measured knowledge document and local artifacts. This example reads an existing result rather than inventing benchmark values:

```python
import json
from pathlib import Path

knowledge = json.loads(Path("measured_knowledge.json").read_text())
result = client.write(
    canonical_ids=addresses,
    session_id="my-run-unique-id",
    knowledge=knowledge,
    files={"patch.diff": "run/patch.diff", "report.md": "run/report.md"},
)
```

The GEAK document shape is:

```text
producer
identity                         # existing producer identity metadata
speedup                          # actual finite measured ratio, when available
value:
  operator                       # optional KernelWiki operator name
  carrier                        # patch (default) or tuned_artifact
  metric:
    bench_key                    # identifies a comparable benchmark/baseline
    metric_kind
    speedup / wall_ms / baseline_wall_ms / case_names
  upstream                       # source repo/commit and framework/precision
  verified_stack / verified_on
  validated / validation_basis
  artifacts                      # logical name -> relative bundle path
```

The client validates local files before uploading and publishes each manifest at every supplied identity rung. GEAK's service does not aggregate prefixes, so a version-independent address needs its own manifest. A write does not promote a champion or mark a result validated. Use an explicit, unique session id for each candidate; retries use the same id.

For tuning tables, use `value.carrier="tuned_artifact"` and provide the table plus any required installation/cache-invalidation metadata. The historical GEAK fields `artifact_files`, `artifact_names`, `apply_env` and `cache_invalidation` can remain in `value` unchanged.

There is no transaction across identity rungs. A failure raises `KBStoreError` naming the completed identities and the retry session id. The client does not silently fall back to a different store or claim a partial write succeeded.

## Search

```python
result = client.search(
    canonical_ids=addresses,
    carrier="patch",
    top_k=5,
    bench_key=knowledge["value"]["metric"]["bench_key"],
)
for candidate in result["candidates"]:
    print(candidate["session_id"], candidate["match"], candidate["speedup"])
```

Search examines the exact page, excludes retired/wrong-carrier/wrong-benchmark entries, and only uses the version fallback when the exact page has no matching candidates. An optional `operator` filter applies to recorded operator metadata; old records without that metadata will not match it.

With a nonempty `bench_key`, matching candidates are ordered by their recorded speedup. Without one, the client does not claim cross-baseline comparability: results are ordered by session id and carry `comparable=False`. Source compatibility and correctness still require validation before applying a patch.

The `scan` argument bounds examined sessions per address (default 200). `scan_truncated=True` means more records may exist outside that bound. Transport/authentication errors raise an exception; an absent page is a normal miss.

The existing `geak:` service supports exact identities, not arbitrary full-text/operator discovery. This interface uses that existing protocol and does not claim to search every remote record by semantic similarity.

## Fetch

```python
candidate = result["candidates"][0]
directory = client.fetch(
    canonical_id=candidate["canonical_id"],
    session_id=candidate["session_id"],
    destination="downloaded_candidate",
)
```

Downloads produce `values.json` and `files/`. The inherited SDK validates relative paths, manifest sizes and SHA-256 digests. The facade also checks that artifacts promised by the knowledge document are present. Downloading does not apply or execute a patch.

## Compatibility and verification

`_store/` contains the Apache-2.0 GEAK protocol implementation with package imports adapted to KernelWiki. `KBStoreClient` remains available for direct access to the original methods. Its identity, local-store, curation, attestation and read-only tests are retained, along with tests of this facade.

The initial validation used local protocol fakes and the inherited offline tests. It did not write a test candidate into the live shared KB service.

## Sources

- [GEAK KB protocol implementation](https://github.com/AMD-AGI/GEAK/tree/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/kb)
- [GEAK kernel experience producer](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/kernel_workflow/scripts/experience_store.py)
