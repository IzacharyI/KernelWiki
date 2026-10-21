import copy
from pathlib import Path

import pytest

from kernel_patch import KBStoreError, PatchClient, kernel_addresses
from kernel_patch._store.store_client import kb_store_url


class MemoryService:
    """Protocol fake: manifests are per identity, just as in the real service."""

    def __init__(self):
        self.records = {}
        self.files = {}
        self.calls = []
        self.fail_id = None

    def put_files(self, cid, sid, items):
        self.calls.append(("files", cid, sid))
        if cid == self.fail_id:
            raise KBStoreError("upload failed")
        self.files[cid, sid] = {rel: Path(path).read_bytes() for rel, path, *_ in items}
        return {rel: "kb://test/" + rel for rel in self.files[cid, sid]}

    def put_knowledge(self, cid, document, *, session_id, mode):
        self.calls.append(("knowledge", cid, session_id))
        self.records[cid, session_id] = copy.deepcopy(document)

    def get_top_sessions(self, cid, *, limit, offset):
        rows = [{"session_id": sid, "score": doc.get("speedup")}
                for (key, sid), doc in self.records.items() if key == cid]
        rows.sort(key=lambda row: -(row.get("score") or 0))
        return {"sessions": rows[offset:offset + limit]}

    def get_session(self, cid, sid):
        doc = self.records.get((cid, sid))
        return {"knowledge": doc} if doc is not None else None

    def download_session(self, cid, sid, destination):
        for rel, data in self.files.get((cid, sid), {}).items():
            target = Path(destination) / "files" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


@pytest.fixture
def addresses():
    return kernel_addresses(kernel_name="_rms_norm_kernel", backend="triton",
                            gfx="gfx950", rocm_version="7.3.1")


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "patch.diff"
    path.write_text("--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n")
    return {"patch.diff": path}


def document(speedup=1.2, bench_key="same-baseline", **value):
    return {"speedup": speedup,
            "value": {"metric": {"bench_key": bench_key}, **value}}


def test_url_alias_precedence_and_whitespace(monkeypatch):
    monkeypatch.setenv("GEAK_KB_URL", "  https://wiki.example  ")
    monkeypatch.setenv("GEAK_KB_STORE_URL", "https://legacy.example")
    monkeypatch.setenv("KB_STORE_URL", "https://fallback.example")
    assert kb_store_url() == "https://wiki.example"
    monkeypatch.setenv("GEAK_KB_URL", "  ")
    assert kb_store_url() == "https://legacy.example"


def test_write_search_fetch_roundtrip_and_all_rung_manifests(addresses, bundle, tmp_path):
    service = MemoryService()
    client = PatchClient(service)
    original = document()
    result = client.write(canonical_ids=addresses, session_id="run-1",
                          knowledge=original, files=bundle)
    assert result["canonical_ids"] == addresses
    assert "artifacts" not in original["value"]
    assert len(service.files) == 2
    hits = client.search(canonical_ids=addresses, bench_key="same-baseline")
    hit, = hits["candidates"]
    assert hit["match"] == "exact" and hit["comparable"]
    destination = client.fetch(canonical_id=hit["canonical_id"], session_id=hit["session_id"],
                               destination=tmp_path / "download")
    assert (destination / "files/patch.diff").read_bytes() == bundle["patch.diff"].read_bytes()
    assert service.calls[0][0] == "files" and service.calls[1][0] == "knowledge"


def test_fallback_is_used_after_filtering_exact_page(addresses):
    service = MemoryService()
    service.records[addresses[0], "wrong-benchmark"] = document(99, "different")
    service.records[addresses[1], "useful"] = document(1.1)
    result = PatchClient(service).search(canonical_ids=addresses, bench_key="same-baseline")
    assert result["candidates"][0]["session_id"] == "useful"
    assert result["candidates"][0]["match"] == "version_fallback"


def test_retired_and_tuned_records_do_not_hide_compatible_patch(addresses):
    service = MemoryService()
    cid = addresses[0]
    service.records[cid, "retired"] = document(9, retained=False)
    service.records[cid, "tuned"] = document(8, carrier="tuned_artifact")
    service.records[cid, "other"] = document(7, "other-baseline")
    service.records[cid, "valid"] = document(1.3)
    result = PatchClient(service).search(canonical_ids=addresses, top_k=1,
                                          bench_key="same-baseline")
    assert [x["session_id"] for x in result["candidates"]] == ["valid"]


def test_without_benchmark_key_does_not_claim_global_ranking(addresses):
    service = MemoryService()
    service.records[addresses[0], "b-fast"] = document(100, "different")
    service.records[addresses[0], "a-slow"] = document(1.1)
    result = PatchClient(service).search(canonical_ids=addresses)
    assert [x["session_id"] for x in result["candidates"]] == ["a-slow", "b-fast"]
    assert not any(x["comparable"] for x in result["candidates"])


def test_partial_failure_reports_completed_rung(addresses, bundle):
    service = MemoryService()
    service.fail_id = addresses[1]
    with pytest.raises(KBStoreError, match="completed identities") as exc:
        PatchClient(service).write(canonical_ids=addresses, session_id="run-1",
                                   knowledge=document(), files=bundle)
    assert addresses[0] in str(exc.value)
    assert (addresses[0], "run-1") in service.records
    assert (addresses[1], "run-1") not in service.records


def test_readonly_and_invalid_artifacts_never_write(monkeypatch, addresses, bundle):
    service = MemoryService()
    monkeypatch.setenv("GEAK_KB_READ_ONLY", "1")
    with pytest.raises(KBStoreError, match="READ_ONLY"):
        PatchClient(service).write(canonical_ids=addresses, session_id="run-1",
                                   knowledge=document(), files=bundle)
    monkeypatch.delenv("GEAK_KB_READ_ONLY")
    with pytest.raises((ValueError, KBStoreError)):
        PatchClient(service).write(canonical_ids=addresses, session_id="run-1",
                                   knowledge=document(), files={"../escape.diff": bundle["patch.diff"]})
    assert service.calls == []


def test_missing_declared_artifact_rejected_on_fetch(addresses, tmp_path):
    service = MemoryService()
    service.records[addresses[0], "broken"] = document(artifacts={"patch": "patch.diff"})
    with pytest.raises(KBStoreError, match="missing declared artifact"):
        PatchClient(service).fetch(canonical_id=addresses[0], session_id="broken",
                                   destination=tmp_path / "download")


def test_pagination_scans_past_first_page_and_reports_bound(addresses):
    service = MemoryService()
    for n in range(205):
        service.records[addresses[0], f"retired-{n}"] = document(9, retained=False)
    service.records[addresses[0], "last-valid"] = document()
    result = PatchClient(service).search(canonical_ids=addresses, scan=300)
    assert result["candidates"][0]["session_id"] == "last-valid"
    assert not result["scan_truncated"]
    bounded = PatchClient(service).search(canonical_ids=addresses, scan=100)
    assert bounded["scan_truncated"] and not bounded["candidates"]


def test_tuned_artifact_roundtrip(addresses, tmp_path):
    table = tmp_path / "config.json"
    table.write_text('{"tile": 64}')
    service = MemoryService()
    client = PatchClient(service)
    client.write(canonical_ids=addresses, session_id="tuned", files={"artifact/config.json": table},
                 knowledge=document(carrier="tuned_artifact", apply_env="CONFIG_PATH"))
    assert not client.search(canonical_ids=addresses)["candidates"]
    result = client.search(canonical_ids=addresses, carrier="tuned_artifact")
    assert result["candidates"][0]["session_id"] == "tuned"


@pytest.mark.parametrize("bench_key", ["", "   ", 42])
def test_invalid_benchmark_filter_is_not_an_unexplained_miss(addresses, bench_key):
    with pytest.raises(ValueError, match="bench_key"):
        PatchClient(MemoryService()).search(canonical_ids=addresses, bench_key=bench_key)
