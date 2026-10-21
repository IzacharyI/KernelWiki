"""Write, discover and download GEAK-compatible patch bundles."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ._store.identity import check, kernel_canonical_ids, kernel_identity
from ._store.store_client import KBStoreClient, KBStoreError, _bundle_rel_path, kb_read_only
from ._store.store_local import validate_session_id


def kernel_addresses(*, kernel_name: str, backend: str, gfx: str,
                     rocm_version: str = "") -> list[str]:
    """Return GEAK's exact and version-independent addresses, in that order.

    kernel_name must be the canonical symbol used by the producer, not an
    extracted filename such as ``triton_<symbol>_task.py``.
    """
    if not all(isinstance(x, str) and x.strip() for x in (kernel_name, backend, gfx)):
        raise ValueError("kernel_name, backend and gfx are required")
    return kernel_canonical_ids(kernel_identity(gfx, kernel_name, backend, rocm_version))


def _addresses(values: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not values:
        raise ValueError("canonical_ids must be a nonempty sequence of addresses")
    return list(dict.fromkeys(check(x) for x in values))


class PatchClient:
    """Small patch interface over the existing KB service; no new server.

    Pass a KBStoreClient explicitly, or use from_env(). Writes never promote a
    champion. Search reports measured candidates, not a global fastest kernel.
    """

    def __init__(self, store: KBStoreClient):
        self.store = store

    @classmethod
    def from_env(cls) -> "PatchClient":
        return cls(KBStoreClient.from_env())

    def write(self, *, canonical_ids: Sequence[str], session_id: str,
              knowledge: dict, files: Mapping[str, str | Path]) -> dict:
        """Publish one session and its files at every supplied identity rung.

        A retry uses the same session_id. If a later rung fails, the exception
        identifies the rungs already completed; there is no cross-rung transaction.
        """
        ids = _addresses(canonical_ids)
        sid = validate_session_id(session_id)
        if kb_read_only():
            raise KBStoreError("write blocked by GEAK_KB_READ_ONLY")
        if not isinstance(knowledge, dict) or not files:
            raise ValueError("knowledge must be an object and files must not be empty")
        document = copy.deepcopy(knowledge)
        json.dumps(document, allow_nan=False)
        value = document.setdefault("value", {})
        if not isinstance(value, dict):
            raise ValueError("knowledge.value must be an object")
        carrier = value.setdefault("carrier", "patch")
        if carrier not in ("patch", "tuned_artifact"):
            raise ValueError("carrier must be patch or tuned_artifact")
        items = []
        for rel, path in files.items():
            rel = _bundle_rel_path(rel)
            path = Path(path)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"artifact is missing or empty: {rel}")
            items.append((rel, path, "rewrite", {}))
        if carrier == "patch" and not any(x[0].endswith((".diff", ".patch")) for x in items):
            raise ValueError("a patch bundle must include a .diff or .patch file")
        promised = value.setdefault("artifacts", {rel: rel for rel in files})
        if not isinstance(promised, dict) or not promised:
            raise ValueError("value.artifacts must be a nonempty object")
        for rel in promised.values():
            if _bundle_rel_path(rel) not in files:
                raise ValueError(f"declared artifact has no file: {rel}")
        completed = []
        references = {}
        for cid in ids:
            try:
                # Commit the manifest before publishing knowledge that promises it.
                references[cid] = self.store.put_files(cid, sid, items)
                if not all(references[cid].get(rel) for rel in files):
                    raise KBStoreError("file commit returned an incomplete manifest")
                self.store.put_knowledge(cid, document, session_id=sid, mode="merge")
                completed.append(cid)
            except Exception as exc:
                raise KBStoreError(
                    f"write failed at {cid}; completed identities: {completed}; "
                    f"retry with session_id={sid}: {exc}"
                ) from exc
        return {"session_id": sid, "canonical_ids": completed, "files": references}

    def search(self, *, canonical_ids: Sequence[str], top_k: int = 5,
               carrier: str = "patch", bench_key: str | None = None,
               operator: str | None = None, scan: int = 200) -> dict:
        """Exact-address discovery with GEAK's version fallback and filtering.

        scan bounds the sessions examined per address, not the final top_k.
        Retired and wrong-carrier records are excluded before top_k is applied.
        With bench_key, only that benchmark's candidates are compared. Without
        it, results are ordered by session id and carry comparable=False.
        """
        ids = _addresses(canonical_ids)
        if carrier not in ("patch", "tuned_artifact"):
            raise ValueError("unsupported carrier")
        if bench_key is not None and (not isinstance(bench_key, str) or not bench_key.strip()):
            raise ValueError("bench_key must be a nonempty string when supplied")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(scan, bool) or not isinstance(scan, int) or not 1 <= scan <= 10000:
            raise ValueError("scan must be between 1 and 10000")
        found, searched = {}, []
        truncated = False
        for cid in ids:
            searched.append(cid)
            offset = 0
            while offset < scan:
                size = min(200, scan - offset)
                try:
                    page = self.store.get_top_sessions(cid, limit=size, offset=offset)
                except KBStoreError as exc:
                    if "HTTP 404" in str(exc):
                        break
                    raise
                rows = (page or {}).get("sessions") or []
                for row in rows:
                    sid = row.get("session_id")
                    if not sid or sid in found:
                        continue
                    record = self.store.get_session(cid, sid)
                    if not isinstance(record, dict):
                        continue
                    doc = record.get("knowledge", record)
                    if not isinstance(doc, dict):
                        continue
                    value = doc.get("value") or {}
                    if not isinstance(value, dict):
                        continue
                    if (value.get("retained") is False or doc.get("retained") is False
                            or value.get("retired_reason") or doc.get("retired_reason")
                            or value.get("lifecycle") == "retracted"):
                        continue
                    if value.get("carrier", "patch") != carrier:
                        continue
                    if operator and value.get("operator", value.get("kernel_class")) != operator:
                        continue
                    metric = value.get("metric") or {}
                    key = metric.get("bench_key") if isinstance(metric, dict) else None
                    key = key or value.get("bench_key") or doc.get("bench_key")
                    if bench_key is not None and key != bench_key:
                        continue
                    score = doc.get("speedup")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        score = None
                    found[sid] = {"canonical_id": cid, "session_id": sid,
                                  "knowledge": doc, "bench_key": key, "speedup": score,
                                  "comparable": bool(bench_key),
                                  "match": "exact" if cid == ids[0] else "version_fallback"}
                offset += len(rows)
                if len(rows) < size:
                    break
                if offset >= scan:
                    truncated = True
            # Prefer a compatible exact page; broaden only if it has no matches.
            if found:
                break
        candidates = list(found.values())
        if bench_key:
            candidates.sort(key=lambda x: (-(x["speedup"] if x["speedup"] is not None
                                               else float("-inf")), x["session_id"]))
        else:
            candidates.sort(key=lambda x: x["session_id"])
        return {"candidates": candidates[:top_k], "searched_ids": searched,
                "scan_truncated": truncated, "matched_count": len(candidates)}

    def fetch(self, *, canonical_id: str, session_id: str,
              destination: str | Path) -> Path:
        """Download values.json and files/, checking manifest sizes and hashes."""
        cid, sid = check(canonical_id), validate_session_id(session_id)
        record = self.store.get_session(cid, sid)
        if not isinstance(record, dict):
            raise KBStoreError("session does not exist")
        doc = record.get("knowledge", record)
        value = doc.get("value", {}) if isinstance(doc, dict) else {}
        artifacts = value.get("artifacts", {}) if isinstance(value, dict) else {}
        if not isinstance(artifacts, dict):
            raise KBStoreError("invalid artifacts document")
        promised = [_bundle_rel_path(rel) for rel in artifacts.values()]
        destination = Path(destination)
        self.store.download_session(cid, sid, destination)
        for rel in promised:
            if not (destination / "files" / rel).is_file():
                raise KBStoreError(f"downloaded bundle is missing declared artifact: {rel}")
        return destination
