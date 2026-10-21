"""Store and retrieve source implementations locally or through GEAK's KB service."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from .bundle import (BundleError, HEX, SourceBundle, applicable, digest, file_names,
                     member, slug, source_fingerprint, split_id, validate_metadata)
from ._store.store_client import KBStoreClient, KBStoreError, kb_read_only

CARRIER = "kernel_source"
INDEX_METRIC = "kernel_source_schema"


def operator_address(operator: str) -> str:
    return "geak:kernel_code:" + slug(operator, "operator")


def implementation_address(implementation_id: str) -> str:
    return "geak:kernel_code:" + ":".join(split_id(implementation_id))


def session_id(implementation_id: str, revision: str) -> str:
    split_id(implementation_id)
    if not isinstance(revision, str) or not HEX.fullmatch(revision):
        raise BundleError("revision must be a complete SHA-256 digest")
    return "kw-" + hashlib.sha256(implementation_id.encode()).hexdigest()[:12] + "-" + revision


@contextmanager
def _snapshot(directory, metadata=None):
    with tempfile.TemporaryDirectory(prefix="kernelwiki-source-") as temporary:
        target = Path(temporary) / "bundle"
        if metadata is None:
            bundle = SourceBundle.load(directory).copy_to(target)
        else:
            data = validate_metadata(metadata)
            root = Path(directory)
            if root.is_symlink() or not root.is_dir():
                raise BundleError("source root must be a directory, not a symlink")
            target.mkdir()
            for rel in file_names(data):
                if rel == "metadata.yaml":
                    continue
                path = target / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(member(root, rel), path)
            (target / "metadata.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            bundle = SourceBundle.load(target)
        yield bundle


def _document(bundle):
    return {INDEX_METRIC: 1, "value": {
        "carrier": CARRIER, "implementation_id": bundle.implementation_id,
        "revision": bundle.revision, "source_digest": bundle.source_digest,
        "metadata": copy.deepcopy(bundle.metadata), "manifest": copy.deepcopy(bundle.manifest),
        "artifact_canonical_id": implementation_address(bundle.implementation_id),
    }}


def _value(record):
    if not isinstance(record, dict):
        raise BundleError("source record is not an object")
    doc = record.get("knowledge", record)
    value = doc.get("value") if isinstance(doc, dict) else None
    if not isinstance(value, dict) or value.get("carrier") != CARRIER:
        raise BundleError("record does not contain a kernel source bundle")
    metadata = validate_metadata(value.get("metadata"))
    if metadata["implementation_id"] != value.get("implementation_id"):
        raise BundleError("record identity disagrees with bundle metadata")
    expected = implementation_address(value["implementation_id"])
    if value.get("artifact_canonical_id") != expected:
        raise BundleError("record artifact address disagrees with its identity")
    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != set(file_names(metadata)):
        raise BundleError("record manifest disagrees with declared bundle files")
    for entry in manifest.values():
        if (not isinstance(entry, dict) or not HEX.fullmatch(str(entry.get("sha256", "")))
                or type(entry.get("size")) is not int or entry["size"] < 0):
            raise BundleError("invalid source manifest entry")
    if digest(manifest) != value.get("revision") or not HEX.fullmatch(str(value.get("source_digest", ""))):
        raise BundleError("invalid source record digests")
    code_digest = source_fingerprint(metadata, manifest)
    if code_digest != value["source_digest"]:
        raise BundleError("source digest disagrees with the recorded runtime files")
    validation = metadata["validation"]
    if validation["status"] == "passed" and validation["source_digest"] != code_digest:
        raise BundleError("validation refers to a different source digest")
    if any(m["kind"] == "measured" and m["source_digest"] != code_digest
           for m in metadata.get("benchmarks", [])):
        raise BundleError("benchmark refers to a different source digest")
    return value


def _summary(value, gfx, dtype, shape, bench_key):
    metadata = value["metadata"]
    measurement = next((m for m in metadata.get("benchmarks", [])
                        if m["kind"] == "measured" and m["gfx"] == gfx and m["dtype"] == dtype
                        and m["shape"] == shape and m["bench_key"] == bench_key), None)
    case = {"gfx": gfx, "dtype": dtype, "shape": shape, "bench_key": bench_key}
    return {"implementation_id": value["implementation_id"], "revision": value["revision"],
            "source_digest": value["source_digest"], "metadata": copy.deepcopy(metadata),
            "measurement": copy.deepcopy(measurement), "comparable": measurement is not None,
            "best_known_for_query": metadata["status"] == "best_known" and measurement is not None
                                    and case in metadata.get("best_for", [])}


def _destination(destination):
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


class KernelClient:
    """Complete source bundles with immutable revisions and explicit evidence.

    Use local(root) for a curated checkout or from_env() for the existing KB
    service. Metadata and source are read without importing or executing kernels.
    """

    def __init__(self, *, root=None, store=None):
        if (root is None) == (store is None):
            raise ValueError("supply exactly one of root or store")
        self.root = Path(root).absolute() if root is not None else None
        self.store = store

    @classmethod
    def local(cls, root="kernels"):
        return cls(root=root)

    @classmethod
    def from_env(cls):
        return cls(store=KBStoreClient.from_env())

    def put(self, source_bundle, metadata=None):
        """Store a complete snapshot; return implementation_id and revision.

        Only explicitly declared files are copied. Existing local entries are
        immutable through this API; edit/review them with Git or choose a new
        implementation name. Remote revisions coexist under the same identity.
        """
        if kb_read_only():
            raise KBStoreError("write blocked by GEAK_KB_READ_ONLY")
        with _snapshot(source_bundle, metadata) as bundle:
            implementation_id = bundle.implementation_id
            result = {"implementation_id": implementation_id, "revision": bundle.revision,
                      "source_digest": bundle.source_digest}
            if self.root is not None:
                target = self.root.joinpath(*split_id(implementation_id))
                for path in (self.root, target.parent.parent, target.parent, target):
                    if path.is_symlink():
                        raise BundleError("collection paths cannot be symlinks")
                if target.exists():
                    if SourceBundle.load(target).revision != bundle.revision:
                        raise FileExistsError("implementation exists with different content; edit it through Git or use a new name")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix=".kernelwiki-", dir=target.parent) as temporary:
                        staged = bundle.copy_to(Path(temporary) / "bundle")
                        if target.exists():
                            raise FileExistsError(target)
                        os.rename(staged.root, target)
                return {**result, "path": str(target)}
            cid = implementation_address(implementation_id)
            sid = session_id(implementation_id, bundle.revision)
            document = _document(bundle)
            items = [(p, bundle.root / p, CARRIER, {}) for p in bundle.manifest]
            completed = []
            try:
                refs = self.store.put_files(cid, sid, items)
                if not all(refs.get(p) for p in bundle.manifest):
                    raise KBStoreError("file commit returned an incomplete manifest")
                self.store.put_knowledge(cid, document, session_id=sid, mode="merge")
                completed.append(cid)
                # The operator page is a discovery index; artifacts have one home.
                index = operator_address(bundle.metadata["operator"])
                self.store.put_knowledge(index, document, session_id=sid, mode="merge")
                completed.append(index)
            except Exception as exc:
                raise KBStoreError(f"source upload failed; completed identities: {completed}; "
                                   f"retry implementation_id={implementation_id}, revision={bundle.revision}: {exc}") from exc
            return {**result, "canonical_id": cid, "session_id": sid}

    def search(self, operator, *, backend=None, gfx=None, dtype=None, shape=None,
               bench_key=None, best_only=False, top_k=20, scan=200):
        """Filter applicability; rank only measurements for an exact benchmark case."""
        slug(operator, "operator")
        if backend is not None:
            slug(backend, "backend")
        if type(top_k) is not int or top_k < 1 or type(scan) is not int or not 1 <= scan <= 10000:
            raise BundleError("top_k must be positive and scan must be between 1 and 10000")
        for name, value in (("gfx", gfx), ("dtype", dtype), ("bench_key", bench_key)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise BundleError(f"{name} must be nonempty when supplied")
        if shape is not None:
            from .bundle import _shape
            _shape(shape, "query shape")
        if (bench_key is not None or best_only) and (not bench_key or not gfx or not dtype or not shape):
            raise BundleError("benchmark ranking and best_only require gfx, dtype, full shape and bench_key")
        found = []
        truncated = False

        def consider(value):
            metadata = value["metadata"]
            if value.get("retained") is False or value.get("retired_reason") or value.get("lifecycle") == "retracted":
                return
            if metadata["operator"] != operator or (backend is not None and metadata["backend"] != backend):
                return
            if not applicable(metadata, gfx, dtype, shape):
                return
            result = _summary(value, gfx, dtype, shape, bench_key)
            if bench_key is not None and not result["comparable"]:
                return
            if best_only and not result["best_known_for_query"]:
                return
            found.append(result)

        if self.root is not None:
            if self.root.is_symlink():
                raise BundleError("collection root cannot be a symlink")
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
            op_root = self.root / operator
            if op_root.is_symlink():
                raise BundleError("collection operator cannot be a symlink")
            for path in sorted(op_root.glob("*/*/metadata.yaml")):
                if path.parent.parent.is_symlink():
                    raise BundleError("collection backend cannot be a symlink")
                bundle = SourceBundle.load(path.parent)
                if path.parent.relative_to(self.root).as_posix() != bundle.implementation_id:
                    raise BundleError("collection path disagrees with implementation_id")
                consider(_document(bundle)["value"])
        else:
            cid = operator_address(operator)
            offset = 0
            seen = set()
            while offset < scan:
                size = min(200, scan - offset)
                try:
                    page = self.store.get_top_sessions(cid, metric=INDEX_METRIC, limit=size, offset=offset)
                except KBStoreError as exc:
                    if "HTTP 404" in str(exc):
                        break
                    raise
                rows = (page or {}).get("sessions") or []
                for row in rows:
                    sid = row.get("session_id")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    record = self.store.get_session(cid, sid)
                    if record is None:
                        continue
                    doc = record.get("knowledge", record)
                    if not isinstance(doc, dict) or (doc.get("value") or {}).get("carrier") != CARRIER:
                        continue
                    if doc.get("retained") is False or doc.get("retired_reason"):
                        continue
                    value = _value(record)
                    if session_id(value["implementation_id"], value["revision"]) != sid:
                        raise BundleError("source index session disagrees with its revision")
                    consider(value)
                offset += len(rows)
                if len(rows) < size:
                    break
                if offset >= scan:
                    truncated = True
        if bench_key and len({digest({k: r["measurement"].get(k) for k in ("baseline", "environment", "protocol")})
                              for r in found}) > 1:
            raise BundleError("benchmark key identifies incompatible baselines, environments or timing protocols")
        found.sort(key=lambda r: ((r["measurement"]["latency_us"] / r["measurement"]["baseline_latency_us"])
                                 if bench_key else 0, r["implementation_id"], r["revision"]))
        return {"implementations": found[:top_k], "matched_count": len(found), "scan_truncated": truncated}

    def fetch(self, implementation_id, *, destination, revision=None):
        """Materialize runnable source files directly into a new directory.

        Pin revision for reproducible retrieval. Without one, remote fetch uses
        the service-selected revision for this implementation (champion/latest).
        """
        parts = split_id(implementation_id)
        if revision is not None:
            session_id(implementation_id, revision)
        target = _destination(destination)
        if self.root is not None:
            source = self.root.joinpath(*parts)
            for path in (self.root, source.parent.parent, source.parent):
                if path.is_symlink():
                    raise BundleError("collection paths cannot be symlinks")
            bundle = SourceBundle.load(source)
            if bundle.implementation_id != implementation_id or (revision is not None and bundle.revision != revision):
                raise BundleError("requested implementation/revision is not in this checkout")
            with tempfile.TemporaryDirectory(prefix=".kernelwiki-", dir=target.parent) as temporary:
                staged = bundle.copy_to(Path(temporary) / "bundle")
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
                os.rename(staged.root, target)
            return target
        cid = implementation_address(implementation_id)
        record = (self.store.get_session(cid, session_id(implementation_id, revision)) if revision is not None
                  else self.store.get_best_record(cid))
        if record is None:
            raise KBStoreError("kernel implementation/revision does not exist")
        value = _value(record)
        if value["implementation_id"] != implementation_id or (revision is not None and value["revision"] != revision):
            raise BundleError("service returned a different implementation/revision")
        sid = session_id(implementation_id, value["revision"])
        if record.get("session_id", sid) != sid:
            raise BundleError("service returned a different session")
        with tempfile.TemporaryDirectory(prefix=".kernelwiki-", dir=target.parent) as temporary:
            download = Path(temporary) / "download"
            self.store.download_session(cid, sid, download, include_values=False)
            bundle = SourceBundle.load(download / "files")
            if (bundle.revision != value["revision"] or bundle.source_digest != value["source_digest"]
                    or bundle.metadata != value["metadata"]):
                raise BundleError("downloaded source does not match recorded metadata and digests")
            staged = bundle.copy_to(Path(temporary) / "bundle")
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            os.rename(staged.root, target)
        return target
