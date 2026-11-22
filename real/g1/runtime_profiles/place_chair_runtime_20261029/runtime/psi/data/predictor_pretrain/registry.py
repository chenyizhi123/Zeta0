"""Machine-readable dataset freezing and cross-stage leakage checks.

The audit is intentionally independent of a particular dataset loader.  Each
adapter emits rows with stable episode/lineage identities; Stage A refuses to
start unless those rows pass this gate against every downstream split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PT_SPLITS = frozenset({"PT_train", "PT_val", "PT_test"})
FT_SPLITS = frozenset({"FT_train", "FT_val", "FT_test"})
DEFAULT_FORBIDDEN_PT_SOURCES = ("psi-data", "humanoidarena")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("encoded lineage must be a JSON list")
            value = parsed
        else:
            value = [stripped]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"expected a sequence of strings, got {type(value).__name__}")
    return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in ("sample_id", "source_id", "revision", "episode_id", "group_id", "split"):
        if key in result and result[key] is not None:
            result[key] = str(result[key]).strip()
    result["lineage_ids"] = list(_as_string_tuple(result.get("lineage_ids")))
    result["content_hashes"] = list(_as_string_tuple(result.get("content_hashes")))
    result["paths"] = list(_as_string_tuple(result.get("paths")))
    return result


def normalize_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical public representation used by manifest hashing."""

    return _normalize_row(row)


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("episodes"))
        if not isinstance(value, list):
            raise ValueError(f"JSON manifest {path} must contain a list or rows/episodes")
        rows = value
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("reading parquet manifests requires pyarrow") from error
        rows = pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"unsupported manifest extension: {path.suffix}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"manifest {path} contains a non-object row")
    return [_normalize_row(row) for row in rows]


def iter_manifest_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream normalized JSONL rows; preserve the existing other-format ABI."""

    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        yield from load_manifest_rows(path)
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest {path} contains a non-object row")
            yield _normalize_row(row)


def manifest_sha256(path: str | Path) -> str:
    """Hash exactly ``canonical_sha256(load_manifest_rows(path))`` incrementally."""

    digest = hashlib.sha256(b"[")
    first = True
    for row in iter_manifest_rows(path):
        if not first:
            digest.update(b",")
        digest.update(canonical_json_bytes(row))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def write_manifest_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write normalized rows; parquet publication is atomic."""

    path = Path(path)
    normalized = sorted(
        (_normalize_row(row) for row in rows),
        key=lambda row: (
            str(row.get("source_id", "")),
            str(row.get("group_id", "")),
            str(row.get("episode_id", "")),
            str(row.get("sample_id", "")),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix.lower() == ".json":
        temporary.write_text(
            json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".jsonl":
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in normalized),
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("writing parquet manifests requires pyarrow") from error
        pq.write_table(pa.Table.from_pylist(normalized), temporary)
    else:
        raise ValueError("output manifest must end in .json, .jsonl or .parquet")
    temporary.replace(path)


def build_grouped_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    """Assign whole capture groups to stable PT splits using a hash ordering."""

    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("split fractions are out of range")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")
    normalized = sorted(
        (_normalize_row(row) for row in rows),
        key=lambda row: (
            str(row.get("source_id", "")),
            str(row.get("group_id", "")),
            str(row.get("episode_id", "")),
            str(row.get("sample_id", "")),
        ),
    )
    for row in normalized:
        if not row.get("group_id"):
            raise ValueError("every pretraining row needs a non-empty group_id")
    groups_by_source: dict[str, set[str]] = {}
    owner_by_group: dict[str, str] = {}
    for row in normalized:
        source, group = str(row.get("source_id", "")), str(row["group_id"])
        if not source:
            raise ValueError("every pretraining row needs a non-empty source_id")
        previous = owner_by_group.setdefault(group, source)
        if previous != source:
            raise ValueError(f"group_id {group!r} spans multiple sources")
        groups_by_source.setdefault(source, set()).add(group)
    assignment = {}
    for source, source_groups in sorted(groups_by_source.items()):
        groups = sorted(
            source_groups,
            key=lambda group: hashlib.sha256(f"{seed}|{source}|{group}".encode()).digest(),
        )
        n = len(groups)
        train_end = round(n * train_fraction)
        val_end = train_end + round(n * val_fraction)
        assignment.update({
            group: ("PT_train" if index < train_end else "PT_val" if index < val_end else "PT_test")
            for index, group in enumerate(groups)
        })
    for row in normalized:
        row["split"] = assignment[row["group_id"]]
    return normalized


@dataclass(frozen=True)
class AuditFinding:
    kind: str
    pt_sample_id: str
    detail: str


@dataclass
class AuditReport:
    format_version: str = "predictor_pretrain_overlap_audit_v1"
    passed: bool = True
    pt_rows: int = 0
    ft_rows: int = 0
    pt_manifest_sha256: str = ""
    ft_manifest_sha256: str = ""
    pt_content_hash_coverage: float = 0.0
    ft_content_hash_coverage: float = 0.0
    findings: list[AuditFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["findings"] = [asdict(item) for item in self.findings]
        return result


class CrossStageAuditAccumulator:
    """Incremental overlap audit that can share another manifest scan."""

    def __init__(
        self,
        ft_rows: Iterable[Mapping[str, Any]],
        *,
        forbidden_pt_roots: Sequence[str | Path] = (),
        forbidden_pt_sources: Sequence[str] = DEFAULT_FORBIDDEN_PT_SOURCES,
        check_duplicate_sample_ids: bool = True,
        progress_every_rows: int = 0,
    ) -> None:
        ft = [_normalize_row(row) for row in ft_rows]
        self.report = AuditReport(
            ft_rows=len(ft),
            ft_manifest_sha256=canonical_sha256(ft),
            ft_content_hash_coverage=(
                sum(bool(row.get("content_hashes")) for row in ft) / len(ft) if ft else 0.0
            ),
        )
        self.findings: list[AuditFinding] = []
        if not ft:
            self.findings.append(AuditFinding("empty_ft_manifest", "<manifest>", "0 rows"))
        self.ft_identity: set[str] = set()
        self.ft_hashes: set[str] = set()
        for index, row in enumerate(ft):
            ft_sample_id = str(row.get("sample_id") or f"ft-row:{index}")
            split = str(row.get("split", ""))
            if split not in FT_SPLITS:
                self.findings.append(
                    AuditFinding("invalid_ft_split", ft_sample_id, split or "<missing>")
                )
            self.ft_identity.update(_as_string_tuple(row.get("lineage_ids")))
            self.ft_hashes.update(_as_string_tuple(row.get("content_hashes")))
            for key in ("sample_id", "episode_id"):
                if row.get(key):
                    self.ft_identity.add(str(row[key]))
        self.roots = tuple(
            Path(root).expanduser().resolve(strict=False) for root in forbidden_pt_roots
        )
        self.forbidden_sources = tuple(value.casefold() for value in forbidden_pt_sources)
        self.groups_by_split: dict[str, set[str]] = {split: set() for split in PT_SPLITS}
        self.identities_db = sqlite3.connect("") if check_duplicate_sample_ids else None
        if self.identities_db is not None:
            self.identities_db.execute("PRAGMA journal_mode=OFF")
            self.identities_db.execute("CREATE TABLE samples(sample_id TEXT PRIMARY KEY)")
        self.pt_digest = hashlib.sha256(b"[")
        self.pt_first = True
        self.pt_hash_rows = 0
        self.progress_every_rows = progress_every_rows
        self.finished = False

    def add(self, raw_row: Mapping[str, Any]) -> None:
        if self.finished:
            raise RuntimeError("cannot add rows to a finished overlap audit")
        row = _normalize_row(raw_row)
        if not self.pt_first:
            self.pt_digest.update(b",")
        self.pt_digest.update(canonical_json_bytes(row))
        self.pt_first = False
        index = self.report.pt_rows
        self.report.pt_rows += 1
        self.pt_hash_rows += int(bool(row.get("content_hashes")))
        sample_id = str(row.get("sample_id") or f"row:{index}")
        if self.identities_db is not None:
            inserted = self.identities_db.execute(
                "INSERT OR IGNORE INTO samples VALUES (?)", (sample_id,)
            ).rowcount
            if not inserted:
                self.findings.append(AuditFinding("duplicate_pt_sample_id", sample_id, sample_id))
        split = str(row.get("split", ""))
        if split not in PT_SPLITS:
            self.findings.append(AuditFinding("invalid_pt_split", sample_id, split or "<missing>"))
        group_id = str(row.get("group_id", ""))
        if split in self.groups_by_split and group_id:
            self.groups_by_split[split].add(group_id)
        source = str(row.get("source_id", ""))
        if any(token in source.casefold() for token in self.forbidden_sources):
            self.findings.append(AuditFinding("forbidden_source", sample_id, source))
        identities = set(_as_string_tuple(row.get("lineage_ids")))
        identities.update(str(row[key]) for key in ("sample_id", "episode_id") if row.get(key))
        overlap = sorted(identities & self.ft_identity)
        if overlap:
            self.findings.append(AuditFinding("ft_lineage_overlap", sample_id, ",".join(overlap)))
        hash_overlap = sorted(set(_as_string_tuple(row.get("content_hashes"))) & self.ft_hashes)
        if hash_overlap:
            self.findings.append(
                AuditFinding("ft_content_hash_overlap", sample_id, ",".join(hash_overlap))
            )
        for raw_path in _as_string_tuple(row.get("paths")):
            candidate = Path(raw_path).expanduser().resolve(strict=False)
            for root in self.roots:
                if candidate == root or root in candidate.parents:
                    self.findings.append(AuditFinding("forbidden_path", sample_id, str(candidate)))
                    break
        if self.progress_every_rows and self.report.pt_rows % self.progress_every_rows == 0:
            print(json.dumps({
                "event": "overlap_audit_progress", "rows": self.report.pt_rows,
                "findings": len(self.findings),
            }), flush=True)

    def close(self) -> None:
        if self.identities_db is not None:
            self.identities_db.close()
            self.identities_db = None

    def finish(self) -> AuditReport:
        if self.finished:
            return self.report
        self.finished = True
        self.close()
        self.pt_digest.update(b"]")
        self.report.pt_manifest_sha256 = self.pt_digest.hexdigest()
        self.report.pt_content_hash_coverage = (
            self.pt_hash_rows / self.report.pt_rows if self.report.pt_rows else 0.0
        )
        if not self.report.pt_rows:
            self.findings.append(AuditFinding("empty_pt_manifest", "<manifest>", "0 rows"))
        split_names = sorted(self.groups_by_split)
        for index, left in enumerate(split_names):
            for right in split_names[index + 1:]:
                for group in sorted(self.groups_by_split[left] & self.groups_by_split[right]):
                    self.findings.append(
                        AuditFinding("pt_group_split_overlap", group, f"{left},{right}")
                    )
        self.report.findings = self.findings
        self.report.passed = not self.findings
        return self.report


def audit_cross_stage_overlap(
    pt_rows: Iterable[Mapping[str, Any]],
    ft_rows: Iterable[Mapping[str, Any]],
    *,
    forbidden_pt_roots: Sequence[str | Path] = (),
    forbidden_pt_sources: Sequence[str] = DEFAULT_FORBIDDEN_PT_SOURCES,
    check_duplicate_sample_ids: bool = True,
    progress_every_rows: int = 0,
) -> AuditReport:
    """Check exact identities plus declared derivation lineage across PT/FT."""
    accumulator = CrossStageAuditAccumulator(
        ft_rows,
        forbidden_pt_roots=forbidden_pt_roots,
        forbidden_pt_sources=forbidden_pt_sources,
        check_duplicate_sample_ids=check_duplicate_sample_ids,
        progress_every_rows=progress_every_rows,
    )
    try:
        for row in pt_rows:
            accumulator.add(row)
        return accumulator.finish()
    finally:
        accumulator.close()
