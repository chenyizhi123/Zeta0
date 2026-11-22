import pytest
import torch
from torch import nn

from psi.models.predictive_world.training import (
    ReferenceModules,
    assert_optimizer_partition,
    local_teacher_alignment,
    module_state_sha256,
)


def test_reference_freeze_covers_parameters_buffers_and_eval_mode():
    reference = ReferenceModules(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2), nn.BatchNorm1d(2))
    before = module_state_sha256(reference)
    component_hashes = reference.freeze()
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert module_state_sha256(reference) == before
    assert set(component_hashes) == {"encoder", "readout", "teacher", "normalizer"}


def test_alignment_freezes_teacher_parameters_but_preserves_latent_gradient():
    teacher = nn.Linear(3, 2, bias=False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    predicted = torch.randn(2, 1, 3, requires_grad=True)
    gold = torch.randn_like(predicted)
    loss, metrics = local_teacher_alignment(
        predicted_latent_white=predicted,
        gold_latent_white=gold,
        unwhiten=lambda value: value,
        teacher_velocity=lambda value: teacher(value),
        action_valid=torch.ones(2, 1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert metrics["alignment_valid_samples"].item() == 2


def test_optimizer_must_own_exact_trainable_partition():
    trainable = nn.Linear(2, 2)
    frozen = nn.Linear(2, 2)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(trainable.parameters())
    assert_optimizer_partition(optimizer, trainable=[trainable], forbidden=[frozen])
    wrong = torch.optim.AdamW(list(trainable.parameters()) + list(frozen.parameters()))
    with pytest.raises(ValueError, match="partition mismatch"):
        assert_optimizer_partition(wrong, trainable=[trainable], forbidden=[frozen])
