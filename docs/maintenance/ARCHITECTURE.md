# Architecture

`predictive_world` is the supported research route. Its pretraining, joint
post-training and deploy-only compositions use the same core modules.
See [the current method](../predictive_world.md).

| Responsibility | Source of truth |
| --- | --- |
| Configuration and loss identity | `src/psi/config/predictive_world.py` |
| Tensor boundaries, indexing, alignment and masks | `src/psi/data/predictive_world/` |
| Frozen image/feature geometry | `src/psi/models/action_aligned_image.py`, `action_aligned_vlm_features.py` |
| Causal conditioning H | `src/psi/models/predictive_world/history.py` |
| Shared state-conditioned block | `src/psi/models/predictive_world/attention.py` |
| Future encoder E/EMA and readout D | `representation.py` in the same directory |
| Prior/world predictor F and normalizer | `latent_transition.py` in the same directory |
| Teacher T and executing policy G | `action_expert.py` (Psi0-derived) and `gr00t.py` (native N1.7 adapter) |
| Host selection and initialization | `factory.py`, `gr00t_assets.py`, validated `host_architecture` |
| Losses, detach boundaries and composition | `phases.py`, `factory.py` in the same directory |
| Checkpoints and explicit transfer | `checkpoint.py`, `initialization.py` in the same directory |
| Safe distributed optimizer updates | `engine.py`, `training.py` in the same directory |
| Training orchestration | `scripts/train/predictive_world/train.py` |
| Deployment and RTC | `src/psi/deploy/predictive_world_history.py`, `predictive_world_service.py` |
| Real observation/action protocols | `src/psi/real_observation_contract.py`, `src/psi/sonic_actions.py` |
| Robot transport and client | `src/psi/deploy/real_sonic_*.py` |

The `predictor_pretrain` data directory retains registry, sampling, state adapters
and converters used by pretraining inputs. It does not provide another model or
trainer. Optional SONIC dependencies use separately pinned external sources.

Dataset scans, full checkpoint hashes and config parsing belong in preparation
or startup. Imports must not initialize CUDA, open sockets or start jobs.

Both hosts use the same objectives and trainer; the original Psi0 baseline under
`src/psi/baselines` is separate. See [backend contracts](../backends.md).
