import pytest
import torch

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank, StateSchemaAdapter


def test_state_schema_adapter_shape_count_and_missing_state_zero():
    adapter = StateSchemaAdapter(29)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 15_808
    assert adapter.expected_parameter_count == 15_808
    raw = torch.randn(2, 8, 29)
    mask = torch.ones_like(raw, dtype=torch.bool)
    present = torch.ones(2, 8, dtype=torch.bool)
    present[1] = False
    mask[1] = False
    output = adapter(raw, mask, state_present=present)
    assert output.shape == (2, 8, 64)
    assert torch.isfinite(output).all()
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_state_schema_adapter_masks_values_and_rejects_ambiguous_missing_rows():
    torch.manual_seed(0)
    adapter = StateSchemaAdapter(3, mean=torch.tensor([1.0, 2.0, 3.0]), std=torch.tensor([1.0, 2.0, 4.0]))
    raw = torch.tensor([[[2.0, 100.0, 7.0]]])
    mask = torch.tensor([[[True, False, True]]])
    changed = raw.clone()
    changed[..., 1] = -1000.0
    torch.testing.assert_close(adapter(raw, mask), adapter(changed, mask))
    with pytest.raises(ValueError):
        adapter(raw, mask, state_present=torch.zeros(1, 1, dtype=torch.bool))


def test_state_adapter_bank_handles_mixed_schemas_and_human_missing_state():
    bank = StateAdapterBank({"hiw29": 29, "he62": 62})
    raw = torch.randn(3, 8, 62)
    mask = torch.ones_like(raw, dtype=torch.bool)
    mask[0, :, 29:] = False
    mask[2] = False
    output = bank(
        raw,
        mask,
        ["hiw29", "he62", "egodex_missing"],
        state_present=torch.tensor([True, True, False]),
    )
    assert output.shape == (3, 8, 64)
    assert torch.equal(output[2], torch.zeros_like(output[2]))
