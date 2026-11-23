# HumanoidArena 7-Task Closed-Loop Evaluation Bridge

This directory provides a standardized closed-loop adapter connecting any causal
Zeta-0 `PredictiveWorldPolicy` (`zeta_native` / Ψ₀ or `gr00t_n1d7` / GR00T N1.7)
to the seven **HumanoidArena v3.1 SONIC Semantic-40** tasks reported in the
paper:

| Task ID | Category | Episodes in `SII-yizhi/HumanoidArenaV3.1_SONIC40` | Frames (50 Hz) |
| --- | --- | ---: | ---: |
| `HOI_double_desk` | Human-Object Interaction | 100 | 102,620 |
| `HOI_football` | Human-Object Interaction | 100 | 56,326 |
| `HOI_pp_box` | Human-Object Interaction | 100 | 69,176 |
| `HSI_boxing` | Human-Scene Interaction | 100 | 55,771 |
| `HSI_open_door` | Human-Scene Interaction | 100 | 83,127 |
| `HSI_sit_sofa` | Human-Scene Interaction | 100 | 70,967 |
| `HSI_vision_navi` | Human-Scene Interaction | 100 | 108,473 |

## 1. Pull the official simulation dataset

```bash
# Preview pull plan for one task (or omit --tasks for all 7 tasks):
zeta data pull humanoidarena --tasks HOI_pp_box --output data/humanoidarena
# Execute download:
zeta data pull humanoidarena --tasks HOI_pp_box --output data/humanoidarena --execute
```

## 2. Connect your HumanoidArena environment to `HumanoidArenaPolicyBridge`

Install [HumanoidArena](https://github.com/William-wAng618/HumanoidArena) in its
simulation environment, then wrap your loaded `PredictiveWorldPolicy`:

```python
from benchmarks.humanoidarena.policy_bridge import (
    HumanoidArenaPolicyBridge,
    write_eval_summary,
)

bridge = HumanoidArenaPolicyBridge(
    policy=policy,
    prepare_qwen_inputs=prepare_inputs_fn,
    action_horizon=40,          # 40 for gr00t_n1d7, 50 for zeta_native
    execute_steps=20,
    state_history_count=1,
    host_architecture="gr00t_n1d7",
    device="cuda:0",
)

metrics = [
    bridge.run_episode(env, task="HOI_pp_box", instruction="Push the box to the goal.", episode_index=i, seed=i)
    for i in range(60)
]
write_eval_summary(metrics, "runs/eval_humanoidarena/summary.json")
```

See [`docs/reproducibility.md`](../../docs/reproducibility.md) for training
preparation and [`scripts/evaluate_predictive_world.py`](../../scripts/evaluate_predictive_world.py)
for strict provenance-bound quality gates.
