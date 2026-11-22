"""Model-independent global sample occurrence streams for scaling runs."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterator, Mapping, Sequence

from .registry import canonical_sha256


COMPACT_V2_ALGORITHM = "weighted_cycle_affine_permutation_v1"


class CompactV2OccurrenceResolver:
    """O(1) deterministic V2 occurrence lookup without a giant stream file.

    One shuffled weighted source cycle preserves the exact requested mixture.
    The ordinal occurrence within that source is then mapped through a keyed
    bijection over its anchors.  Every source anchor is therefore visited once
    before that source starts its next independently permuted epoch.
    """

    def __init__(
        self,
        *,
        source_weights: Mapping[str, int],
        source_capacities: Mapping[str, int],
        seed: int,
        state_dropout_probability: float,
    ) -> None:
        if not 0.0 <= state_dropout_probability <= 1.0:
            raise ValueError("state dropout probability must be in [0,1]")
        weights = {str(key): int(value) for key, value in source_weights.items()}
        capacities = {str(key): int(value) for key, value in source_capacities.items()}
        if not weights or set(weights) != set(capacities):
            raise ValueError("source weights and capacities must have the same nonempty keys")
        if any(value <= 0 for value in weights.values()) or any(
            value <= 0 for value in capacities.values()
        ):
            raise ValueError("source weights and capacities must be positive")
        cycle = [source for source in sorted(weights) for _ in range(weights[source])]
        random.Random(int(seed)).shuffle(cycle)
        seen = {source: 0 for source in weights}
        ranks = []
        for source in cycle:
            ranks.append(seen[source])
            seen[source] += 1
        self.source_weights = weights
        self.source_capacities = capacities
        self.seed = int(seed)
        self.state_dropout_probability = float(state_dropout_probability)
        self._cycle = tuple(cycle)
        self._cycle_source_ranks = tuple(ranks)

    @staticmethod
    def _permuted_position(*, size: int, position: int, epoch: int, seed: int, source: str) -> int:
        if size == 1:
            return 0
        digest = hashlib.sha256(
            f"{COMPACT_V2_ALGORITHM}|{seed}|{source}|{epoch}".encode()
        ).digest()
        multiplier = int.from_bytes(digest[:8], "big") % size
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(digest[8:16], "big") % size
        return (multiplier * position + offset) % size

    def source_and_position(self, occurrence_index: int) -> tuple[str, int]:
        occurrence_index = int(occurrence_index)
        if occurrence_index < 0:
            raise ValueError("occurrence_index must be non-negative")
        cycle_number, cycle_position = divmod(occurrence_index, len(self._cycle))
        source = self._cycle[cycle_position]
        source_ordinal = (
            cycle_number * self.source_weights[source]
            + self._cycle_source_ranks[cycle_position]
        )
        capacity = self.source_capacities[source]
        epoch, position = divmod(source_ordinal, capacity)
        return source, self._permuted_position(
            size=capacity,
            position=position,
            epoch=epoch,
            seed=self.seed,
            source=source,
        )

    def occurrence(self, occurrence_index: int) -> dict[str, Any]:
        source, source_position = self.source_and_position(occurrence_index)
        return {
            "occurrence_index": int(occurrence_index),
            "source_id": source,
            "source_position": source_position,
            "drop_state": occurrence_state_dropout(
                seed=self.seed,
                occurrence_index=int(occurrence_index),
                probability=self.state_dropout_probability,
            ),
        }


def occurrence_state_dropout(*, seed: int, occurrence_index: int, probability: float) -> bool:
    """Stateless dropout decision, deliberately independent of model RNG."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("dropout probability must be in [0,1]")
    digest = hashlib.sha256(f"state-dropout-v1|{seed}|{occurrence_index}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    return uniform < probability


def iter_sample_occurrences(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_weights: Mapping[str, int],
    total_updates: int,
    global_batch_size: int,
    seed: int,
    state_dropout_probability: float = 0.2,
    sampling_version: str = "v1",
) -> Iterator[dict[str, Any]]:
    """Yield the authoritative pre-sharding stream shared by every model size.

    Integer source weights avoid floating-point boundary differences.  The
    generated artifact, rather than a rank-local sampler, is the experimental
    source of truth.
    """

    if total_updates <= 0 or global_batch_size <= 0:
        raise ValueError("total_updates and global_batch_size must be positive")
    if sampling_version not in {"v1", "v2", "v3"}:
        raise ValueError("sampling_version must be v1, v2 or v3")
    if not source_weights or any(int(weight) <= 0 for weight in source_weights.values()):
        raise ValueError("source weights must be positive integers")
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "PT_train":
            continue
        source = str(row.get("source_id", ""))
        sample_id = str(row.get("sample_id", ""))
        if not source or not sample_id:
            raise ValueError("PT_train rows require source_id and sample_id")
        by_source[source].append(row)
    if set(by_source) != set(source_weights):
        raise ValueError(
            f"source weights and PT_train sources differ: {sorted(source_weights)} vs {sorted(by_source)}"
        )
    for source in by_source:
        by_source[source].sort(key=lambda row: str(row["sample_id"]))

    sources = sorted(source_weights)
    if sampling_version in {"v2", "v3"}:
        resolver = CompactV2OccurrenceResolver(
            source_weights=source_weights,
            source_capacities={source: len(by_source[source]) for source in sources},
            seed=seed,
            state_dropout_probability=state_dropout_probability,
        )
        for occurrence_index in range(total_updates * global_batch_size):
            occurrence = resolver.occurrence(occurrence_index)
            source = occurrence.pop("source_id")
            source_position = occurrence.pop("source_position")
            row = by_source[source][source_position]
            yield {
                **occurrence,
                "update": occurrence_index // global_batch_size,
                "position_in_global_batch": occurrence_index % global_batch_size,
                "source_id": source,
                "sample_id": str(row["sample_id"]),
            }
        return
    cumulative: list[tuple[int, str]] = []
    running = 0
    for source in sources:
        running += int(source_weights[source])
        cumulative.append((running, source))
    total_weight = running
    rng = random.Random(seed)
    for occurrence_index in range(total_updates * global_batch_size):
        draw = rng.randrange(total_weight)
        source = next(source for boundary, source in cumulative if draw < boundary)
        source_rows = by_source[source]
        row = source_rows[rng.randrange(len(source_rows))]
        yield {
            "occurrence_index": occurrence_index,
            "update": occurrence_index // global_batch_size,
            "position_in_global_batch": occurrence_index % global_batch_size,
            "source_id": source,
            "sample_id": str(row["sample_id"]),
            "drop_state": occurrence_state_dropout(
                seed=seed,
                occurrence_index=occurrence_index,
                probability=state_dropout_probability,
            ),
        }


def sample_stream_spec_sha256(
    *,
    dataset_manifest_sha256: str,
    source_weights: Mapping[str, int],
    total_updates: int,
    global_batch_size: int,
    seed: int,
    state_dropout_probability: float,
    sampling_version: str = "v1",
) -> str:
    payload = {
        "format_version": f"predictor_pretrain_sample_stream_{sampling_version}",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_weights": dict(sorted(source_weights.items())),
        "total_updates": total_updates,
        "global_batch_size": global_batch_size,
        "seed": seed,
        "state_dropout_probability": state_dropout_probability,
    }
    if sampling_version in {"v2", "v3"}:
        payload["sampling_algorithm"] = COMPACT_V2_ALGORITHM
    return canonical_sha256(payload)


__all__ = [
    "COMPACT_V2_ALGORITHM",
    "CompactV2OccurrenceResolver",
    "iter_sample_occurrences",
    "occurrence_state_dropout",
    "sample_stream_spec_sha256",
]
