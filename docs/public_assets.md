# Public source and external asset catalog

Use this page to find the code, models and hardware dependencies for your setup.
The [quickstart](quickstart.md) runs on synthetic data; training and deployment
require the assets listed below. Task-trained Zeta checkpoints are not currently
published.

## Exact public code

From the repository root, preview and install the G1 source dependencies:

```bash
python scripts/fetch_sources.py GR00T-WholeBodyControl
python scripts/fetch_sources.py GR00T-WholeBodyControl --apply
python scripts/setup_g1_dependencies.py
python scripts/setup_g1_dependencies.py --apply
```

Source revisions are listed in [sources.toml](../third_party/sources.toml).
The SDK and geometry selection is defined in
[PUBLIC_DEPENDENCIES.json](../real/g1/PUBLIC_DEPENDENCIES.json).
See [source verification](maintenance/RELEASE_CHECKLIST.md#source-checks) for release checks.

## Models, data and hardware runtime

| Asset | Where to get it | Setup requirement |
| --- | --- | --- |
| HumanoidArena 7-task dataset | [`SII-yizhi/HumanoidArenaV3.1_SONIC40`](https://huggingface.co/datasets/SII-yizhi/HumanoidArenaV3.1_SONIC40) | Public LeRobot SONIC semantic-40 (50 Hz, 700 episodes); pull with `zeta data pull humanoidarena` |
| Real-world Unitree G1 dataset | [`SII-yizhi/zeta0-real-dataset`](https://huggingface.co/datasets/SII-yizhi/zeta0-real-dataset) | Public Unitree G1 + BrainCo LeRobot v3 (739 episodes, 6+1 tasks); pull with `zeta data pull real` (or `--tasks place_chair`) |
| Psi0 base models and data | [Psi0](https://github.com/physical-superintelligence-lab/Psi0), [models](https://huggingface.co/USC-PSI-Lab/psi-model), [data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data) | Match the encoder and processor in your config |
| GR00T N1.7 base | [NVIDIA model card](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Record the model revision and follow its license terms |
| SONIC controller and models | [WBC installation](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) | Match ONNX weights, observation config and controller build |
| BrainCo Revo2 SDK | [SDK documentation](https://www.brainco-hz.com/docs/revolimb-hand/revo2/parameters.html), [BrainCoTech SDK](https://github.com/BrainCoTech/brainco-hand-sdk) | Obtain compatible C libraries and headers from the supplier |
| Robot-side BrainCo DDS service | [Unitree service](https://github.com/unitreerobotics/brainco_hand_service), [BrainCo integration](https://github.com/BrainCoTech/unitree-g1-brainco-hand) | Install a compatible dual-hand DDS service separately on the robot |
| PICO app and PC service | [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) | Install the runtime for your workstation architecture |
| Demonstrations | Your recordings or a licensed dataset | Record task labels, timestamps, schemas and splits |
| Task-trained Zeta weights and first frames | Your task exports and demonstration references | Match the task, encoder and deployment runtime |

Component licenses and vendor redistribution restrictions are listed in
[third-party notices](../THIRD_PARTY_NOTICES.md) and the
[G1 license review](maintenance/G1_INTEGRATION_REVIEW.md).

## Real-world G1 dataset (`SII-yizhi/zeta0-real-dataset`)

The real-world Unitree G1 + SONIC + BrainCo release contains **739 episodes (872,409 frames, ~24.4 GB)** packaged as per-task ZIP archives with `SHA256SUMS` (three 640×480 AV1 RGB cameras: `ego_view`, `left_wrist`, `right_wrist`).

| Task Archive | Split | Episodes | Rate | Schema / Preparation Script |
| --- | --- | :---: | :---: | --- |
| `real/place_chair.zip` | Primary | 101 | 30 Hz | Flat `state[41]` / `action[76]` · `prepare_sonic_lerobot_task.py` |
| `real/blow_dry_hair.zip` | Primary | 81 | 30 Hz | Flat `state[41]` / `action[76]` · `prepare_sonic_lerobot_task.py` |
| `real/place_tape_on_table.zip` | Primary | 100 | 30 Hz | Flat `state[41]` / `action[76]` · `prepare_sonic_lerobot_task.py` |
| `real/clean_table_trash.zip` | Primary | 86 | 30 Hz | Flat `state[41]` / `action[76]` · `prepare_sonic_lerobot_task.py` |
| `real/tidy_scene.zip` | Primary | 200 | 30 Hz | Flat `state[41]` / `action[76]` · `prepare_sonic_lerobot_task.py` |
| `real/place_doll.zip` | Primary | 100 *(ep 80 excluded)* | 50 Hz | Split `state[29]+hand_q[12]` / `motion_token[64]+hand_q[12]` + timing sidecars · `prepare_real_sonic_task.py` |
| `additional/flatten_box.zip` | Additional | 71 | 50 Hz | Split `state[29]+hand_q[12]` / `motion_token[64]+hand_q[12]` + timing sidecars · `prepare_real_sonic_task.py` |

Verify downloaded archives with `sha256sum -c SHA256SUMS`, unzip the task directory, audit it with `zeta data inspect <task_dir> --full`, and prepare training runs with `--init-mode scratch` or `--init-mode gr00t_n1d7`.

## Capture-to-training boundary

Use the native `sonic_g1` adapter for official LeRobot v3 recordings. Historical
v2.1 recipes use a different loader. Before training, check state/action columns,
normalization, timestamps, episode quality and splits with
[data inspection](framework.md#inspect-captured-data), then follow
[real-data preparation](real_data_final_method.md).
