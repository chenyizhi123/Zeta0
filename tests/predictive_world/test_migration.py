import pytest
import torch
from torch import nn

from psi.models.predictive_world.migration import migrate_state_dict


def test_migration_requires_exact_mapping_shape_and_allowlists():
    target = nn.Linear(2, 3)
    source = {"old.w": torch.ones_like(target.weight), "old.b": torch.zeros_like(target.bias)}
    report = migrate_state_dict(
        source, {"module": target},
        {"old.w": "module.weight", "old.b": "module.bias"},
    )
    assert len(report.copied) == 2 and torch.equal(target.weight, source["old.w"])
    with pytest.raises(ValueError, match="initialized target allowlist"):
        migrate_state_dict(source, {"module": nn.Linear(2, 3)}, {"old.w": "module.weight"},
                           allow_unused_sources=frozenset({"old.b"}))


def test_migration_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        migrate_state_dict(
            {"old": torch.zeros(1)}, {"module": nn.Linear(2, 3)},
            {"old": "module.weight"}, allow_initialized_targets=frozenset({"module.bias"}),
        )
