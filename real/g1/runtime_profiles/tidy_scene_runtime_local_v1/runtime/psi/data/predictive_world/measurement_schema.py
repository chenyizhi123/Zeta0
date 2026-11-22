"""Audited physical-field registry for future body supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SINGLE_CAMERA_ALIGNMENT = "single_camera"
SHARED_FRAME_INDEX_ALIGNMENT = "shared_frame_index"
INDEPENDENT_CAMERA_ALIGNMENT = "independent_camera_physical_time"


@dataclass(frozen=True)
class CanonicalBodyMapping:
    """Audited source-native -> shared body-coordinate mapping.

    ``None`` raw indices are real missing channels.  They are materialized as
    zero with validity false; they are never imputed or interpreted as a
    physical zero.  Values are standardized with source statistics so the
    shared slots have one optimization scale while retaining an explicit mask.
    """

    layout_id: str
    slots: tuple[str, ...]
    raw_indices: tuple[int | None, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    normalization_space: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_width: int
    ) -> "CanonicalBodyMapping":
        import math

        raw = value.get("raw_indices")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("canonical_body.raw_indices must be an array")
        indices = tuple(
            None if item is None else int(item)
            for item in raw
        )
        result = cls(
            layout_id=str(value.get("layout_id", "")),
            slots=tuple(str(item) for item in value.get("slots", ())),
            raw_indices=indices,
            mean=tuple(float(item) for item in value.get("mean", ())),
            std=tuple(float(item) for item in value.get("std", ())),
            normalization_space=str(value.get("normalization_space", "")),
        )
        width = len(result.slots)
        if (
            not result.layout_id
            or "REQUIRED_" in result.layout_id
            or width <= 0
            or len(set(result.slots)) != width
            or any(not item or "REQUIRED_" in item for item in result.slots)
            or len(result.raw_indices) != width
            or len(result.mean) != width
            or len(result.std) != width
            or not result.normalization_space
            or "REQUIRED_" in result.normalization_space
        ):
            raise ValueError("canonical_body mapping is incomplete")
        present = [item for item in result.raw_indices if item is not None]
        if (
            len(present) != len(set(present))
            or any(item < 0 or item >= source_width for item in present)
        ):
            raise ValueError("canonical_body raw indices are duplicated or out of range")
        if any(
            not math.isfinite(mean)
            or not math.isfinite(std)
            or std <= 0
            for mean, std in zip(result.mean, result.std)
        ):
            raise ValueError("canonical_body statistics must be finite with positive std")
        return result

    @property
    def signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                slot,
                1,
                self.layout_id,
                self.normalization_space,
            )
            for slot in self.slots
        )


class FieldKind(str, Enum):
    MEASUREMENT = "measurement"
    COMMAND = "command"
    DERIVED = "derived"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MeasurementField:
    name: str
    start: int
    stop: int
    kind: FieldKind
    unit: str | None
    coordinate_frame: str | None
    rotation_representation: str | None
    time_key: str
    statistics_id: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeasurementField":
        required = {"name", "start", "stop", "kind", "time_key"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"measurement field is missing keys: {missing}")
        result = cls(
            name=str(value["name"]),
            start=int(value["start"]),
            stop=int(value["stop"]),
            kind=FieldKind(str(value["kind"])),
            unit=None if value.get("unit") is None else str(value["unit"]),
            coordinate_frame=(
                None if value.get("coordinate_frame") is None else str(value["coordinate_frame"])
            ),
            rotation_representation=(
                None
                if value.get("rotation_representation") is None
                else str(value["rotation_representation"])
            ),
            time_key=str(value["time_key"]),
            statistics_id=(
                None if value.get("statistics_id") is None else str(value["statistics_id"])
            ),
        )
        if not result.name or result.start < 0 or result.stop <= result.start or not result.time_key:
            raise ValueError("measurement field name/slice/time_key is invalid")
        if "REQUIRED_" in result.time_key:
            raise ValueError("measurement field time_key is unresolved")
        return result

    @property
    def eligible_body_target(self) -> bool:
        return (
            self.kind == FieldKind.MEASUREMENT
            and bool(self.unit)
            and "REQUIRED_" not in str(self.unit)
            and bool(self.statistics_id)
            and "REQUIRED_" not in str(self.statistics_id)
        )


@dataclass(frozen=True)
class MeasurementSchema:
    schema_id: str
    width: int
    fields: tuple[MeasurementField, ...]
    world_action_eligible: bool
    canonical_body: CanonicalBodyMapping | None = None

    @classmethod
    def from_mapping(cls, schema_id: str, value: Mapping[str, Any]) -> "MeasurementSchema":
        fields = tuple(MeasurementField.from_mapping(item) for item in value.get("fields", ()))
        canonical = value.get("canonical_body")
        result = cls(
            schema_id=str(schema_id),
            width=int(value["width"]),
            fields=fields,
            world_action_eligible=bool(value.get("world_action_eligible", False)),
            canonical_body=(
                None
                if canonical is None
                else CanonicalBodyMapping.from_mapping(
                    canonical, source_width=int(value["width"])
                )
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.schema_id or self.width <= 0 or not self.fields:
            raise ValueError("measurement schema must have an ID, positive width and fields")
        names: set[str] = set()
        occupied: set[int] = set()
        for field in self.fields:
            if field.name in names:
                raise ValueError(f"duplicate field name {field.name!r}")
            names.add(field.name)
            if field.stop > self.width:
                raise ValueError(f"field {field.name!r} exceeds schema width")
            indices = set(range(field.start, field.stop))
            if occupied & indices:
                raise ValueError(f"field {field.name!r} overlaps another field")
            occupied |= indices

    @property
    def body_target_fields(self) -> tuple[MeasurementField, ...]:
        return tuple(field for field in self.fields if field.eligible_body_target)

    @property
    def body_target_indices(self) -> tuple[int, ...]:
        if self.canonical_body is not None:
            return tuple(
                index for index in self.canonical_body.raw_indices if index is not None
            )
        return tuple(
            index
            for field in self.body_target_fields
            for index in range(field.start, field.stop)
        )

    @property
    def body_target_signature(self) -> tuple[tuple[Any, ...], ...]:
        """Canonical physical/normalization meaning after gathering body targets."""

        if self.canonical_body is not None:
            return self.canonical_body.signature

        return tuple(
            (
                field.name,
                field.stop - field.start,
                field.unit,
                field.coordinate_frame,
                field.rotation_representation,
                field.statistics_id,
                field.time_key,
            )
            for field in self.body_target_fields
        )


@dataclass(frozen=True)
class SourceCapability:
    """Audited source-level eligibility; absence is never interpreted as support."""

    source_id: str
    measurement_schema_id: str
    action_protocol_id: str
    visual_keys: tuple[str, ...]
    future_visual_keys: tuple[str, ...]
    supports_future_visual: bool
    supports_future_measurement: bool
    supports_world_action: bool
    audit_revision: str
    shared_frame_index_across_visual_keys: bool
    independent_timestamps_across_visual_keys: bool
    visual_synchronization_audit_revision: str | None

    @classmethod
    def from_mapping(cls, source_id: str, value: Mapping[str, Any]) -> "SourceCapability":
        flag_names = (
            "supports_future_visual",
            "supports_future_measurement",
            "supports_world_action",
            "shared_frame_index_across_visual_keys",
            "independent_timestamps_across_visual_keys",
        )
        for flag_name in flag_names:
            if flag_name in value and not isinstance(value[flag_name], bool):
                raise ValueError(
                    f"source {source_id!r} capability {flag_name} must be a JSON boolean"
                )
        raw_visual_keys = value.get("visual_keys", ())
        if not isinstance(raw_visual_keys, (list, tuple)):
            raise ValueError(f"source {source_id!r} visual_keys must be a JSON array")
        result = cls(
            source_id=str(source_id),
            measurement_schema_id=str(value.get("measurement_schema_id", "")),
            action_protocol_id=str(value.get("action_protocol_id", "")),
            visual_keys=tuple(str(item) for item in raw_visual_keys),
            future_visual_keys=tuple(
                str(item)
                for item in value.get("future_visual_keys", raw_visual_keys)
            ),
            supports_future_visual=bool(value.get("supports_future_visual", False)),
            supports_future_measurement=bool(value.get("supports_future_measurement", False)),
            supports_world_action=bool(value.get("supports_world_action", False)),
            audit_revision=str(value.get("audit_revision", "")),
            shared_frame_index_across_visual_keys=bool(
                value.get("shared_frame_index_across_visual_keys", False)
            ),
            independent_timestamps_across_visual_keys=bool(
                value.get("independent_timestamps_across_visual_keys", False)
            ),
            visual_synchronization_audit_revision=(
                None
                if value.get("visual_synchronization_audit_revision") is None
                else str(value["visual_synchronization_audit_revision"])
            ),
        )
        if (
            not result.source_id
            or not result.measurement_schema_id
            or not result.action_protocol_id
            or not result.visual_keys
            or not result.audit_revision
            or "REQUIRED_" in result.measurement_schema_id
            or "REQUIRED_" in result.action_protocol_id
            or "REQUIRED_" in result.audit_revision
            or any(not key or "REQUIRED_" in key for key in result.visual_keys)
            or len(set(result.visual_keys)) != len(result.visual_keys)
            or not result.future_visual_keys
            or len(set(result.future_visual_keys)) != len(result.future_visual_keys)
            or not set(result.future_visual_keys).issubset(result.visual_keys)
        ):
            raise ValueError(f"source {source_id!r} has incomplete capability provenance")
        if result.shared_frame_index_across_visual_keys and (
            not result.visual_synchronization_audit_revision
            or "REQUIRED_" in result.visual_synchronization_audit_revision
        ):
            raise ValueError(
                f"source {source_id!r} claims shared visual frame indices without "
                "a synchronization audit revision"
            )
        if (
            result.shared_frame_index_across_visual_keys
            and result.independent_timestamps_across_visual_keys
        ):
            raise ValueError(
                f"source {source_id!r} cannot declare both shared and independent "
                "multi-camera alignment"
            )
        if len(result.visual_keys) <= 1 and (
            result.shared_frame_index_across_visual_keys
            or result.independent_timestamps_across_visual_keys
            or result.visual_synchronization_audit_revision is not None
        ):
            raise ValueError(
                f"source {source_id!r} declares shared-camera synchronization without "
                "multiple visual keys"
            )
        if (
            not result.shared_frame_index_across_visual_keys
            and result.visual_synchronization_audit_revision is not None
        ):
            raise ValueError(
                f"source {source_id!r} has a synchronization audit without enabling "
                "shared frame indices"
            )
        return result

    def index_contract(self) -> dict[str, Any]:
        """Return the exact capability snapshot frozen into every index row."""

        if len(self.visual_keys) > 1 and not (
            self.shared_frame_index_across_visual_keys
            or self.independent_timestamps_across_visual_keys
        ):
            raise ValueError(
                f"source {self.source_id!r} has multiple visual keys but the current index "
                "ABI requires an explicitly audited shared-index or independent-time "
                "alignment mode"
            )
        if len(self.visual_keys) == 1:
            mode = SINGLE_CAMERA_ALIGNMENT
        elif self.shared_frame_index_across_visual_keys:
            mode = SHARED_FRAME_INDEX_ALIGNMENT
        else:
            mode = INDEPENDENT_CAMERA_ALIGNMENT
        return {
            "source_id": self.source_id,
            "measurement_schema_id": self.measurement_schema_id,
            "action_protocol_id": self.action_protocol_id,
            "visual_keys": list(self.visual_keys),
            "future_visual_keys": list(self.future_visual_keys),
            "supports_future_visual": self.supports_future_visual,
            "supports_future_measurement": self.supports_future_measurement,
            "supports_world_action": self.supports_world_action,
            "audit_revision": self.audit_revision,
            "visual_alignment_mode": mode,
            "visual_synchronization_audit_revision": (
                self.visual_synchronization_audit_revision
                if mode == SHARED_FRAME_INDEX_ALIGNMENT
                else None
            ),
        }


def canonical_body_target_signature(
    capabilities: Mapping[str, SourceCapability],
    measurement_schemas: Mapping[str, MeasurementSchema],
) -> tuple[tuple[Any, ...], ...]:
    """Require one physical and normalization meaning across all indexed sources."""

    reference: tuple[tuple[Any, ...], ...] | None = None
    reference_source = ""
    for source_id in sorted(capabilities):
        capability = capabilities[source_id]
        schema = measurement_schemas[capability.measurement_schema_id]
        signature = schema.body_target_signature
        if not signature:
            raise ValueError(
                f"source {source_id!r} has no audited canonical body target fields"
            )
        if reference is None:
            reference, reference_source = signature, source_id
        elif signature != reference:
            raise ValueError(
                "future measurement schemas do not share one canonical physical/"
                f"normalization signature: {reference_source!r} != {source_id!r}"
            )
    if reference is None:
        raise ValueError("capability registry has no canonical body target signature")
    return reference


def canonical_body_target_width(signature: Sequence[Sequence[Any]]) -> int:
    return sum(int(field[1]) for field in signature)


def assert_canonical_body_target_width(
    capabilities: Mapping[str, SourceCapability],
    measurement_schemas: Mapping[str, MeasurementSchema],
    *,
    expected_width: int,
) -> tuple[tuple[Any, ...], ...]:
    """Bind the model measurement width to the audited gather/statistics ABI."""

    signature = canonical_body_target_signature(capabilities, measurement_schemas)
    actual_width = canonical_body_target_width(signature)
    if actual_width != int(expected_width):
        raise ValueError(
            "model measurement_dim disagrees with the canonical registry body target "
            f"width: expected {expected_width}, registry {actual_width}"
        )
    return signature


def validate_predictive_world_row_capability(
    row: Mapping[str, Any],
    *,
    capabilities: Mapping[str, SourceCapability],
    measurement_schemas: Mapping[str, MeasurementSchema],
) -> tuple[SourceCapability, MeasurementSchema]:
    """Fail closed when one immutable row exceeds its audited source contract."""

    source_id = str(row.get("source_id", ""))
    if source_id not in capabilities:
        raise ValueError(f"index source {source_id!r} is absent from capability registry")
    capability = capabilities[source_id]
    schema = measurement_schemas[capability.measurement_schema_id]
    expected_contract = capability.index_contract()
    if row.get("source_capability") != expected_contract:
        raise ValueError(
            f"index source capability snapshot disagrees with registry for {source_id!r}"
        )
    materialization = row.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ValueError("index row lacks audited materialization metadata")
    if (
        str(materialization.get("measurement_schema_id")) != capability.measurement_schema_id
        or str(materialization.get("action_protocol_id")) != capability.action_protocol_id
    ):
        raise ValueError("index materialization disagrees with source capability registry")
    videos = materialization.get("video_paths")
    if not isinstance(videos, Mapping) or set(map(str, videos)) != set(capability.visual_keys):
        raise ValueError("index materialization visual keys disagree with source capability registry")
    raw_measurement_indices = materialization.get("measurement_indices")
    if not isinstance(raw_measurement_indices, list) or any(
        isinstance(value, bool) or (value is not None and not isinstance(value, int))
        for value in raw_measurement_indices
    ):
        raise ValueError(
            "materialization requires integer-or-null canonical measurement indices"
        )
    expected_indices: tuple[int | None, ...]
    if schema.canonical_body is not None:
        expected_indices = schema.canonical_body.raw_indices
    else:
        expected_indices = schema.body_target_indices
    if tuple(raw_measurement_indices) != expected_indices:
        raise ValueError(
            "materialization measurement_indices must exactly equal the registry mapping"
        )
    for mask_name in (
        "future_visual_valid", "future_measurement_valid", "world_action_group_valid"
    ):
        mask = row.get(mask_name)
        if not isinstance(mask, list) or any(not isinstance(value, bool) for value in mask):
            raise ValueError(f"index row {mask_name} must be an explicit boolean list")
    if any(bool(value) for value in row.get("future_visual_valid", ())) and not capability.supports_future_visual:
        raise ValueError("index grants future visual supervision to an unsupported source")
    if (
        any(bool(value) for value in row.get("future_measurement_valid", ()))
        and not capability.supports_future_measurement
    ):
        raise ValueError("index grants future measurement supervision to an unsupported source")
    if (
        any(bool(value) for value in row.get("world_action_group_valid", ()))
        and not capability.supports_world_action
    ):
        raise ValueError("index grants world-action supervision to an unsupported source")
    return capability, schema


def load_measurement_registry(path: str | Path) -> dict[str, MeasurementSchema]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("measurement registry must be a non-empty JSON object")
    return {
        str(schema_id): MeasurementSchema.from_mapping(str(schema_id), value)
        for schema_id, value in payload.items()
    }


def load_capability_registry(
    path: str | Path,
    *,
    measurement_schemas: Mapping[str, MeasurementSchema],
    action_protocol_ids: set[str] | frozenset[str],
) -> dict[str, SourceCapability]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("capability registry must be a non-empty JSON object")
    result = {
        str(source_id): SourceCapability.from_mapping(str(source_id), value)
        for source_id, value in payload.items()
    }
    for source_id, capability in result.items():
        if capability.measurement_schema_id not in measurement_schemas:
            raise ValueError(f"source {source_id!r} references an unknown measurement schema")
        if capability.action_protocol_id not in action_protocol_ids:
            raise ValueError(f"source {source_id!r} references an unknown action protocol")
        schema = measurement_schemas[capability.measurement_schema_id]
        if capability.supports_future_measurement and not schema.body_target_signature:
            raise ValueError(f"source {source_id!r} claims future measurements without eligible fields")
        if capability.supports_world_action and not schema.world_action_eligible:
            raise ValueError(f"source {source_id!r} claims world-action support forbidden by its schema")
    canonical_body_target_signature(result, measurement_schemas)
    return result


__all__ = [
    "CanonicalBodyMapping",
    "FieldKind",
    "INDEPENDENT_CAMERA_ALIGNMENT",
    "MeasurementField",
    "MeasurementSchema",
    "SHARED_FRAME_INDEX_ALIGNMENT",
    "SINGLE_CAMERA_ALIGNMENT",
    "SourceCapability",
    "assert_canonical_body_target_width",
    "canonical_body_target_signature",
    "canonical_body_target_width",
    "load_capability_registry",
    "load_measurement_registry",
    "validate_predictive_world_row_capability",
]
