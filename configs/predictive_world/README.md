# Predictive-world configuration

`src/psi/config/predictive_world.py` validates configuration. Defaults and legacy
fields remain compatible with recorded pretraining artifacts. Choose the explicit
profile matching the experiment; loading an old template does not select the final
method automatically.

| Template | Purpose |
| --- | --- |
| `world_model.template.json` | Base pretraining world stage and normalizer calibration |
| `joint.template.json` | Base joint action-pretraining stage |
| `joint.monitoring-v1.template.json` | Base joint stage with opt-in diagnostics |
| `quality_gates.template.json` | Explicit evaluation thresholds |

Templates contain `REQUIRED_` placeholders. Task preparation supplies immutable
registries/index/cache bindings and verified source migrations. The final profile
initializes all modules from a complete joint-pretrained checkpoint; it does not
reset the normalizer or silently reuse optimizer state.

Real G1/BrainCo uses a state41/motion76 data contract, RTC15, four ranks
and 20,000 updates. See [the real-data recipe](../../docs/real_data_final_method.md).
Resource layouts are recorded in resolved configs, not encoded in model layers.

See [reproducibility](../../docs/reproducibility.md) for environment/setup and
[the method contract](../../docs/predictive_world.md) for gradient paths and masks.
