# Psi0-derived Zeta

[`joint.template.json`](joint.template.json) is the portable v6/raw joint recipe.
It selects `model.host_architecture=zeta_native`, the historical serialized name
of Zeta's Psi0-derived host, and `training.joint_from_scratch=true`. Only the VLM
is pretrained; Zeta task modules start randomly.

Use [the backend workflow](../../docs/backends.md) and
[real-data preparation](../../docs/reproducibility.md#inputs-and-configuration). Replace
all `REQUIRED_` inputs with the preparation tools. The original Psi0 **baseline**
uses a separate adapter.

Historical pretraining/migration templates remain under
[`configs/predictive_world`](../predictive_world/README.md).
