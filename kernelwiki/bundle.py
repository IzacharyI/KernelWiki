"""Validate and fingerprint complete, reproducible kernel source bundles."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import yaml

SCHEMA_VERSION = 1
SOURCE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cu", ".hip", ".h", ".hpp", ".s", ".S", ".mlir"}
GROUPS = {"source", "tests", "benchmarks", "config", "support"}
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class BundleError(ValueError):
    """A source bundle or query does not meet the published contract."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise BundleError(f"metadata keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def canonical_json(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise BundleError("metadata must contain finite JSON-compatible values") from exc


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def slug(value, field="identifier") -> str:
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise BundleError(f"{field} must be a lowercase alphanumeric/underscore/hyphen slug")
    return value


def split_id(implementation_id: str) -> tuple[str, str, str]:
    if not isinstance(implementation_id, str) or len(implementation_id.split("/")) != 3:
        raise BundleError("implementation_id must be operator/backend/name")
    return tuple(slug(part) for part in implementation_id.split("/"))


def relative_path(value) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\0" in value or any(p in ("", ".", "..") for p in value.split("/"))):
        raise BundleError(f"invalid bundle-relative path: {value!r}")
    if Path(value).suffix.lower() in (".patch", ".diff"):
        raise BundleError("source bundles do not accept patch or diff artifacts")
    return value


def member(root: Path, rel: str) -> Path:
    rel = relative_path(rel)
    path = root
    for part in rel.split("/"):
        path = path / part
        if path.is_symlink():
            raise BundleError(f"bundle members cannot be symlinks: {rel}")
    if not path.is_file():
        raise BundleError(f"missing bundle file: {rel}")
    return path


def _strings(value, field, *, nonempty=True):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(x, str) or not x.strip() for x in value)):
        raise BundleError(f"{field} must be a list of nonempty strings")


def _shape(shape, field):
    if (not isinstance(shape, dict) or not shape
            or any(not isinstance(k, str) or not k or isinstance(v, bool)
                   or not isinstance(v, int) or v < 0 for k, v in shape.items())):
        raise BundleError(f"{field} must map dimension names to nonnegative integers")


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise BundleError(f"{field} must be finite and positive")


def _evidence(value, field, declared):
    _strings(value, field)
    for reference in value:
        if re.match(r"^(?:https?|kb|s3|gs|run):", reference):
            continue
        if relative_path(reference) not in declared:
            raise BundleError(f"{field}: local evidence must be a declared bundle file: {reference}")


def validate_metadata(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise BundleError("metadata must be an object")
    data = copy.deepcopy(raw)
    canonical_json(data)
    if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise BundleError("schema_version must be 1")
    for field in ("operator", "backend", "name"):
        slug(data.get(field), field)
    expected = "/".join(data[x] for x in ("operator", "backend", "name"))
    if data.get("implementation_id") != expected:
        raise BundleError(f"implementation_id must be {expected}")
    if data.get("status") not in ("reference", "validated", "best_known"):
        raise BundleError("status must be reference, validated or best_known")
    for field in ("description", "license"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise BundleError(f"{field} is required")
    if not isinstance(data.get("provenance"), dict) or not data["provenance"]:
        raise BundleError("provenance is required")
    _strings(data.get("dependencies"), "dependencies", nonempty=False)
    files = data.get("files")
    if not isinstance(files, dict) or set(files) - GROUPS:
        raise BundleError("files must contain source/tests/benchmarks/config/support lists")
    seen = {"metadata.yaml"}
    for group in GROUPS:
        names = files.get(group, [])
        _strings(names, f"files.{group}", nonempty=group in ("source", "tests", "benchmarks"))
        for rel in names:
            relative_path(rel)
            if rel in seen:
                raise BundleError(f"file is declared more than once: {rel}")
            seen.add(rel)
    if "README.md" not in seen:
        raise BundleError("declare README.md in files.support")
    if not any(Path(p).suffix in SOURCE_SUFFIXES for p in files["source"]):
        raise BundleError("files.source must include an actual source file")
    entry = data.get("entry_point")
    if not isinstance(entry, dict) or entry.get("file") not in files["source"]:
        raise BundleError("entry_point.file must be a declared source file")
    if not isinstance(entry.get("symbol"), str) or not entry["symbol"].strip():
        raise BundleError("entry_point.symbol is required")
    commands = data.get("commands", {})
    if not isinstance(commands, dict):
        raise BundleError("commands must be an object")
    for field in ("test", "benchmark"):
        _strings(commands.get(field), f"commands.{field}")
    supports = data.get("supports", {})
    if not isinstance(supports, dict):
        raise BundleError("supports must be an object")
    for field in ("gfx", "dtypes"):
        _strings(supports.get(field), f"supports.{field}")
    rules = supports.get("shape")
    if not isinstance(rules, dict) or not rules:
        raise BundleError("supports.shape must declare supported dimensions")
    for dimension, rule in rules.items():
        if not isinstance(dimension, str) or not dimension or not isinstance(rule, dict) or not rule:
            raise BundleError("shape rules must be nonempty objects")
        if set(rule) - {"min", "max", "multiple_of", "values"}:
            raise BundleError(f"unsupported shape rule for {dimension}")
        for key, value in rule.items():
            values = value if key == "values" else [value]
            if (not isinstance(values, list) or not values or any(type(x) is not int or x < 0 for x in values)
                    or (key == "multiple_of" and value == 0)):
                raise BundleError(f"invalid shape rule for {dimension}.{key}")
        if rule.get("min", 0) > rule.get("max", float("inf")):
            raise BundleError("shape minimum exceeds maximum")
    validation = data.get("validation", {})
    if not isinstance(validation, dict):
        raise BundleError("validation must be an object")
    if validation.get("status") not in ("not_run", "passed"):
        raise BundleError("validation.status must be not_run or passed")
    if validation["status"] == "passed":
        _evidence(validation.get("evidence"), "validation.evidence", seen)
        if not HEX.fullmatch(str(validation.get("source_digest", ""))):
            raise BundleError("passed validation must identify its source_digest")
    if data["status"] != "reference" and validation["status"] != "passed":
        raise BundleError("validated/best_known status requires passed validation")
    benchmarks = data.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        raise BundleError("benchmarks must be a list")
    contexts = set()
    for measurement in benchmarks:
        if not isinstance(measurement, dict) or measurement.get("kind") not in ("measured", "upstream_reported"):
            raise BundleError("benchmark kind must be measured or upstream_reported")
        _evidence(measurement.get("evidence"), "benchmark evidence", seen)
        if measurement["kind"] != "measured":
            continue
        for field in ("gfx", "dtype", "bench_key", "baseline"):
            if not isinstance(measurement.get(field), str) or not measurement[field].strip():
                raise BundleError(f"measured benchmark requires {field}")
        _shape(measurement.get("shape"), "benchmark.shape")
        if not applicable(data, measurement["gfx"], measurement["dtype"], measurement["shape"]):
            raise BundleError("measured benchmark lies outside declared supports")
        if set(measurement["shape"]) != set(rules):
            raise BundleError("measured benchmark must specify every shape dimension")
        for field in ("latency_us", "baseline_latency_us"):
            _number(measurement.get(field), field)
        if (measurement.get("correctness") != "passed" or not isinstance(measurement.get("environment"), dict)
                or not measurement["environment"]):
            raise BundleError("measured benchmark requires passed correctness and its environment")
        if not HEX.fullmatch(str(measurement.get("source_digest", ""))):
            raise BundleError("measured benchmark requires source_digest")
        context = canonical_json({k: measurement[k] for k in ("gfx", "dtype", "shape", "bench_key")})
        if context in contexts:
            raise BundleError("duplicate measured benchmark context")
        contexts.add(context)
    best_for = data.get("best_for", [])
    if not isinstance(best_for, list):
        raise BundleError("best_for must be a list")
    if data["status"] == "best_known" and not best_for:
        raise BundleError("best_known must identify measured best_for cases")
    for case in best_for:
        if not isinstance(case, dict) or set(case) != {"gfx", "dtype", "shape", "bench_key"}:
            raise BundleError("best_for cases require gfx/dtype/shape/bench_key")
        matching = [m for m in benchmarks if m["kind"] == "measured"
                    and all(m[k] == case[k] for k in case)]
        if not matching or matching[0]["latency_us"] >= matching[0]["baseline_latency_us"]:
            raise BundleError("best_for requires a measured improvement for that exact case")
    return data


def applicable(metadata, gfx=None, dtype=None, shape=None):
    supports = metadata["supports"]
    if gfx is not None and gfx not in supports["gfx"]:
        return False
    if dtype is not None and dtype not in supports["dtypes"]:
        return False
    if shape is not None:
        _shape(shape, "query shape")
        for dimension, value in shape.items():
            rule = supports["shape"].get(dimension)
            if rule is None or value < rule.get("min", 0) or value > rule.get("max", float("inf")):
                return False
            if "values" in rule and value not in rule["values"]:
                return False
            if value % rule.get("multiple_of", 1):
                return False
    return True


def file_names(metadata):
    return sorted({"metadata.yaml"} | {p for names in metadata["files"].values() for p in names})


def source_fingerprint(metadata, manifest):
    runtime = sorted(metadata["files"]["source"] + metadata["files"].get("config", []))
    return digest({"files": {p: manifest[p] for p in runtime},
                   "entry_point": metadata["entry_point"], "dependencies": metadata["dependencies"]})


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    metadata: dict
    manifest: dict
    revision: str
    source_digest: str

    @property
    def implementation_id(self):
        return self.metadata["implementation_id"]

    @classmethod
    def load(cls, directory: str | Path):
        root = Path(directory)
        if root.is_symlink() or not root.is_dir():
            raise BundleError("bundle root must be a directory, not a symlink")
        try:
            metadata = validate_metadata(yaml.load(member(root, "metadata.yaml").read_text(), Loader=_UniqueLoader))
        except yaml.YAMLError as exc:
            raise BundleError("invalid metadata.yaml") from exc
        manifest = {}
        for rel in file_names(metadata):
            content = member(root, rel).read_bytes()
            if rel in metadata["files"]["source"]:
                if not content or content.startswith(b"diff --git "):
                    raise BundleError(f"not a complete source file: {rel}")
            manifest[rel] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        source_digest = source_fingerprint(metadata, manifest)
        validation = metadata["validation"]
        if validation["status"] == "passed" and validation["source_digest"] != source_digest:
            raise BundleError("validation belongs to different source code or runtime configuration")
        for measurement in metadata.get("benchmarks", []):
            if measurement["kind"] == "measured" and measurement["source_digest"] != source_digest:
                raise BundleError("benchmark belongs to different source code or runtime configuration")
        return cls(root.resolve(), metadata, manifest, digest(manifest), source_digest)

    def copy_to(self, destination: str | Path):
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=False)
        try:
            for rel in self.manifest:
                path = target / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(member(self.root, rel), path)
            copied = SourceBundle.load(target)
            if copied.revision != self.revision:
                raise BundleError("source bundle changed while copying")
            return copied
        except Exception:
            shutil.rmtree(target)
            raise
