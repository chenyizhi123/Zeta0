# ζ₀ (Zeta0)

**A Framework for World Action Models with Joint Scene and Body Foresight in Humanoid Control**

[![Project Page](https://img.shields.io/badge/Project_Page-ζ₀-0F766E?style=flat-square)](https://chenyizhi123.github.io/zeta0-project/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_(18pp)-B91C1C?style=flat-square)](https://chenyizhi123.github.io/zeta0-project/assets/Zeta0-Humanoid-Full.pdf)
[![Sim Dataset](https://img.shields.io/badge/HF_Dataset-HumanoidArena_SONIC40-F59E0B?style=flat-square)](https://huggingface.co/datasets/SII-yizhi/HumanoidArenaV3.1_SONIC40)
[![Real Dataset](https://img.shields.io/badge/HF_Dataset-Zeta0_Real_G1-F59E0B?style=flat-square)](https://huggingface.co/datasets/SII-yizhi/zeta0-real-dataset)
[![License](https://img.shields.io/badge/License-Apache_2.0-2563EB?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10_train_|_3.12_capture-3776AB?style=flat-square)](docs/installation.md)

English | [简体中文](README_zh.md) · [Project Page](https://chenyizhi123.github.io/zeta0-project/) · [Paper PDF](https://chenyizhi123.github.io/zeta0-project/assets/Zeta0-Humanoid-Full.pdf) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

**Zeta0** turns layered / dual-system vision-language-action (VLA) policies into **World Action Models (WAM)** by foreseeing future visual features and measured whole-body joint trajectories in a shared latent representation, supervised in action space via **dual-response transfer**. It connects Unitree G1 + SONIC + BrainCo demonstration capture, unified training across **Ψ₀** and **GR00T N1.7** hosts (with **π₀.₅** host contracts reserved), closed-loop evaluation, foresight visualization, and **Real-Time Chunking (RTC)** deployment.

![Zeta-0 policy: autonomous G1 task, scene-and-body prediction, and additional real-world tasks](docs/assets/zeta0-overview.png)

*Zeta-0 foresees joint scene-and-body futures (+0.2 / +0.4 / +0.6 s) from causal inputs to condition whole-body humanoid actions. See the [method contract](docs/predictive_world.md).*

---

## Key Results

| Benchmark / Metric | Base Policy | **ζ₀ (Ours)** | Gain | Details |
| --- | :---: | :---: | :---: | --- |
| **HumanoidArena (7-task mean, Ψ₀)** | 44.8% | **63.6%** | **+18.8 pp** | 420 episodes/setting · `zeta_native` |
| **HumanoidArena (7-task mean, GR00T N1.7)** | 70.5% | **79.5%** | **+9.0 pp** | Tops all compared 7-task joint policies · `gr00t_n1d7` |
| **Real-World Unitree G1 (6 tasks, Ψ₀)** | 61.7% | **78.3%** | **+16.7 pp** | 300 physical trials · +26.7 pp on multi-stage tasks |
| **Real-World Unitree G1 (6 tasks, GR00T N1.7)** | 75.0% | **85.0%** | **+10.0 pp** | SONIC whole-body control + BrainCo hands + RTC |
| **29-Joint Body Forecast (+0.6 s RMSE)** | 9.36° (hold) | **4.53°** | **−51.6%** | Wins in 83.5% of 526 windows (legs: 5.23° vs 12.54°) |
| **Closed-Loop Inference (RTX 4090)** | — | **212 ms** | **6.3 GiB** | Single deterministic forecast per action chunk |

---

## Layered Architecture (`System 2 → 1.5 → 1 → 0`)

Zeta0 sits as a causal **System 1.5 Foresight Bridge** between a frozen System 2 VLM and a System 1 flow-matching action expert, unified across training, export, and deployment via the Layered VLA Host Registry (`zeta0.hosts` / `psi.models.predictive_world.hosts`):

| Layer | Component | Implementations |
| --- | --- | --- |
| **System 2 (VLM)** | Observation & visual patch backbone | Frozen Qwen3-VL (`zeta_native`), Exported GR00T N1.7 VLM (`gr00t_n1d7`), PaliGemma + KV-cache (`pi05`, planned) |
| **System 1.5 (ζ₀ Core)** | Shared foresight & action-space transfer | `E_φ / E_φ̄` + `WorldReadout D` (training only) · `H` + `F_θ(prior/world)` · Dual-Response Transfer `L_resp` |
| **System 1 (Action Expert)** | Flow-matching policy + RTC inpainting | `TemporalNativeActionExpert` (`zeta_native`), native `Gr00tN1d7ActionHead` (`gr00t_n1d7`), Gemma expert (`pi05`, planned) |
| **System 0 (Robot Control)** | Whole-body execution on Unitree G1 | Unified G1 Runtime (`unified`) · RTC chunk handoff · GEAR-SONIC · BrainCo Revo2 hands |

> At deployment, `E_φ`, `E_φ̄`, `D`, Teacher `T_ψ`, and world-mode future actions are excluded by construction inside `PredictiveWorldPolicy`.

---

## Four Open-Source Real-Robot Pillars (Unitree G1 + SONIC + BrainCo)

This release centers on an end-to-end, reproducible real-humanoid stack across four pillars:

| Pillar | What's Included | Entrypoints |
| --- | --- | --- |
| **1. Hardware Kit (3D-Printable)** | RealSense D415 wrist mounts (paper `+30°` back-of-hand & expanded `−20°..+20°` palm-side, recommended `0°`), print specs, M3 BOM, URDF previews & SHA-256 asset manifest | [`hardware/README.md`](hardware/README.md) · [`hardware/assets_manifest.json`](hardware/assets_manifest.json) |
| **2. Real G1 Collection** | PICO / XRoboToolkit teleop + GEAR-SONIC WBC + BrainCo Revo2 dual-hand + 3× D415 capture directly into LeRobot v3 (`state41` / `motion76`), plus offline audit (`zeta data inspect`) | [`real/g1/README.md`](real/g1/README.md#collection) · [`docs/framework.md`](docs/framework.md#inspect-captured-data) |
| **3. Real-Robot Training** | No-private-weights v6/raw training from public **Qwen3-VL / Cosmos-Reason2-2B** (`--init-mode scratch`) or **NVIDIA GR00T N1.7-3B** (`--init-mode gr00t_n1d7`, 30 Hz & 50 Hz) on `zeta0-real-dataset` or your own G1 recordings | [`scripts/train/predictive_world/prepare_sonic_lerobot_task.py`](scripts/train/predictive_world/prepare_sonic_lerobot_task.py) · [`docs/real_data_final_method.md`](docs/real_data_final_method.md) |
| **4. Real G1 Deployment** | Unified G1 runtime (`unified`) + `real_sonic_server` supporting both `zeta_native` (`50/30` & `30/50`) and `gr00t_n1d7` (`30/40` & `50/40`) with RTC inpainting + hardware-free `mock_g1_client.py` + `zeta visualize log` | [`real/g1/runtime_profiles/README.md`](real/g1/runtime_profiles/README.md) · [`real/g1/README.md`](real/g1/README.md#deployment) |

### Train on Real G1 Data (No Private Checkpoints Required)

```bash
# Option A: Train Zeta-0 (Ψ₀ / zeta_native) from scratch with any public Qwen3-VL-2B backbone on 30 Hz G1 data
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode scratch --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model /path/to/Qwen3-VL-2B --output runs/place_chair_scratch \
  --current-state-only --fixed-action-grid --whole-body-target --action-horizon 50 \
  --gpus 1 --batch 64 --updates 20000

# Option B: Train Zeta-0 on native GR00T N1.7-3B (30 Hz / 40-step action chunk) and deploy directly to G1
PYTHONPATH=src python scripts/train/gr00t_n1d7/export_backbone.py \
  --base /path/to/GR00T-N1.7-3B --cosmos-processor /path/to/Cosmos-Reason2-2B --output assets/gr00t_n1d7_backbone
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode gr00t_n1d7 --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model assets/gr00t_n1d7_backbone --gr00t-model /path/to/GR00T-N1.7-3B \
  --output runs/place_chair_gr00t --current-state-only --fixed-action-grid --whole-body-target \
  --gpus 1 --batch 64 --updates 20000

# Then cache visual features, train, export, and serve (append --gr00t-model for gr00t_n1d7)
PYTHONPATH=src python scripts/data/cache_predictive_world_features.py \
  --index runs/place_chair_gr00t/artifacts/predictive_world_index.jsonl \
  --provenance runs/place_chair_gr00t/artifacts/feature_provenance.json \
  --model assets/gr00t_n1d7_backbone --output runs/place_chair_gr00t/artifacts/feature_cache
PYTHONPATH=src python scripts/train/predictive_world/train.py --config runs/place_chair_gr00t/config.json
PYTHONPATH=src python scripts/export_real_sonic_research_policy.py \
  --checkpoint runs/place_chair_gr00t/joint/joint-final \
  --config runs/place_chair_gr00t/config.json --output runs/place_chair_gr00t/deploy_pkg
PYTHONPATH=src python -m psi.deploy.real_sonic_server \
  --weights-dir runs/place_chair_gr00t/deploy_pkg \
  --qwen-model assets/gr00t_n1d7_backbone --gr00t-model /path/to/GR00T-N1.7-3B
```

---

## Released Source & Public Datasets

**2026-11-23 · v0.2.0.** Includes the public `zeta0` Python SDK facade, unified Layered VLA Host Registry (training + export + G1 deployment for both `zeta_native` and `gr00t_n1d7`), no-private-weights real-robot training (`--init-mode scratch` and `--init-mode gr00t_n1d7` across 30 Hz and 50 Hz G1 tasks), full-algorithm CPU quickstart with visual report, `zeta visualize` foresight/RTC dashboard, HumanoidArena evaluation bridge, hardware-free G1 mock client, official Dockerfiles, and offline CI suites.

| Asset | Source | Format / Specs |
| --- | --- | --- |
| **Real-World G1 Dataset (6+1 Tasks)** | [`SII-yizhi/zeta0-real-dataset`](https://huggingface.co/datasets/SII-yizhi/zeta0-real-dataset) | Unitree G1 + BrainCo + SONIC · 739 episodes · 872,409 rows (~24.4 GB, task ZIPs + `SHA256SUMS`) · 5× 30 Hz flat (`sonic_lerobot`) + 2× 50 Hz split (`real_sonic`) |
| **Simulation Dataset (7 Tasks)** | [`SII-yizhi/HumanoidArenaV3.1_SONIC40`](https://huggingface.co/datasets/SII-yizhi/HumanoidArenaV3.1_SONIC40) | Public · LeRobot · 700 episodes · 546,460 frames · 50 Hz · `state64` / `semantic40` |
| **Base Models & Hardware SDKs** | [Asset Catalog](docs/public_assets.md) | Qwen3-VL / Ψ₀ base, NVIDIA GR00T N1.7-3B, GEAR-SONIC WBC, BrainCo SDK, XRoboToolkit |
| **Task-Trained Weights** | [Artifact Requirements](docs/artifacts.md) | External assets (not bundled in this source repo) |

---

## Quick Start (30 Seconds on CPU, No Hardware or Downloads)

```bash
git clone https://github.com/chenyizhi123/Zeta0.git
cd Zeta0

# 1. Create lightweight CPU environment (Python 3.10)
uv venv --python 3.10 .venv-check
uv pip install --python .venv-check/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-check/bin/python --group cpu-test

# 2. Run full-algorithm Zeta-0 demo (E/EMA + D + F prior/world + T/G Dual-Response + Causal Deploy + Plot)
make demo PYTHON=.venv-check/bin/python
```

The demo trains a compact full Zeta-0 stack on synthetic kinematic trajectories, checkpoints at update 4, verifies bit-exact reload and resume through update 8, validates causal `PredictiveWorldPolicy` deployment, and writes both `summary.json` and **`foresight_preview.png`** under `runs/quickstart-*`.

### Public Python SDK (`import zeta0`)

```python
import zeta0
from zeta0.core import (
    FutureTrajectoryEncoder,
    WorldReadout,
    TemporalLatentTransition,
    JointObjective,
    PredictiveWorldPolicy,
)
from zeta0.hosts import list_hosts, describe_hosts

print(zeta0.__version__, list_hosts())
# 0.2.0 ('gr00t_n1d7', 'pi05', 'zeta_native')
```

---

## End-to-End Operations with `zeta` CLI

```bash
# Inspect environments and supported layered VLA hosts
PYTHONPATH=src python -m psi.cli doctor --profile train --python /path/to/python
PYTHONPATH=src python -m psi.cli hosts

# Pull official simulation (single task or all 7) or real G1 dataset (plan by default; add --execute to download)
zeta data pull humanoidarena --tasks HOI_pp_box --output data/humanoidarena
zeta data pull real --output data/zeta0-real

# Scaffold a v6/raw training config without hand-editing template JSONs
zeta init --host gr00t_n1d7 --qwen-model /path/to/backbone --gr00t-model /path/to/GR00T-N1.7-3B --output configs/runs/pp_box.json

# Visualize foresight rollouts or G1 RuntimeAudit RTC latency traces
zeta visualize rollout runs/quickstart-*/summary.json --output runs/foresight.png
zeta visualize log runs/mock_g1_trace.jsonl --output runs/rtc_dashboard.png

# Hardware-free Sim-in-the-Loop G1 RTC dry-run (no physical robot required)
PYTHONPATH=src python scripts/deploy/mock_g1_client.py --in-process --cycles 12 --output runs/mock_g1_trace.jsonl

# Benchmark deploy-only foresight + action-expert latency (CPU or CUDA, optional --compile)
PYTHONPATH=src python scripts/benchmark_inference.py --preset tiny-cpu --device cpu
```

---

## Workflows: Collection → Training → Evaluation → Deployment

| Goal | Guide & Entrypoints |
| --- | --- |
| **Install / Docker** | [Installation](docs/installation.md) · [Dockerfiles](docker/README.md) · [Environment check](docs/framework.md#start-with-your-existing-environment) |
| **Collect G1 Demonstrations** | [G1 Capture & Teleop](real/g1/README.md#collection) · [Data inspection](docs/framework.md#inspect-captured-data) |
| **Train (`Ψ₀` or `GR00T N1.7`)** | [Layered Backends](docs/backends.md) · [Reproducibility](docs/reproducibility.md) · [Ψ₀ recipe](configs/psi0/README.md) · [GR00T N1.7 recipe](configs/gr00t_n1d7/README.md) |
| **Evaluate in HumanoidArena** | [HumanoidArena 7-Task Bridge](benchmarks/humanoidarena/README.md) |
| **Deploy on Unitree G1** | [Unified G1 Runtime](real/g1/runtime_profiles/README.md) · [Real SONIC Service](real/README.md) · [G1 Deployment](real/g1/README.md#deployment) |

| Model Host | `model.host_architecture` | Status | Default Grid / Solver | Config Template |
| --- | --- | :---: | :---: | --- |
| **Ψ₀-derived** | `zeta_native` | Supported (Train + Export + G1 Deploy) | 50 steps / 10 Euler | [`configs/psi0/joint.template.json`](configs/psi0/joint.template.json) |
| **Native GR00T N1.7** | `gr00t_n1d7` | Supported (Train + Export + G1 Deploy) | 40 steps / 4 Euler | [`configs/gr00t_n1d7/joint.template.json`](configs/gr00t_n1d7/joint.template.json) |
| **π₀.₅ (PaliGemma + Gemma Expert)** | `pi05` | Contract Reserved (`kv_cache` ready) | 50 steps / 10 Euler | [`src/psi/models/predictive_world/hosts/pi05.py`](src/psi/models/predictive_world/hosts/pi05.py) |

---

## Open-Source Wrist Camera Mounts (Unitree G1 + RealSense D415)

The [`hardware/`](hardware/README.md) kit documents 3D-printable wrist mounts,
printing specs, BOM guidelines, and URDF previews. Heavy `.stl` meshes and
simulation FOV videos are distributed via the verified release bundle
(`hardware-wrist-camera-v1.tar.gz`, indexed in [`hardware/assets_manifest.json`](hardware/assets_manifest.json))
to keep `git clone` fast and lightweight.

| Mount | Camera Side | Angles | URDF Preview |
| --- | --- | --- | --- |
| Paper mount | Back of hand | `+30°` | [30°](hardware/wrist-camera-paper/urdf_30deg.png) |
| Expanded mount *(recommended: **0°**)* | Palm side | `−20°`, `−10°`, **`0°`**, `+10°`, `+20°` | [−20°](hardware/wrist-camera-expansion/urdf_neg20deg.png) · [−10°](hardware/wrist-camera-expansion/urdf_neg10deg.png) · [**0°**](hardware/wrist-camera-expansion/urdf_0deg.png) · [10°](hardware/wrist-camera-expansion/urdf_10deg.png) · [20°](hardware/wrist-camera-expansion/urdf_20deg.png) |

---

## Citation

If you find Zeta0 useful in your research or robot deployment, please cite:

```bibtex
@article{chen2026zeta0,
  title   = {$\zeta_0$: A Framework for World Action Models with Joint Scene and Body Foresight in Humanoid Control},
  author  = {Chen, Yizhi and Cao, Zhanxiang and Wang, Yitong and Liu, Junyu and Zeng, Hongchu and Peng, Xinyi and Huang, Chuxuan and Su, Yicheng and Zhou, Zihao and Fu, Shengcheng and Li, Haoyang and Yan, Liyun and Zhan, Tianyue and Ma, Jianming and Zhang, Yang and Zhu, Keqi and Si, Xiaxi and Yao, Jinming and Sun, Qiao and Zhang, Shizeng and Zhong, Jiangwei and Gao, Yue},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://chenyizhi123.github.io/zeta0-project/}
}
```

---

## Documentation, Acknowledgments & Licenses

[Documentation Index](docs/README.md) · [中文文档导航](docs/README_zh.md) · [CLI Guide](docs/framework.md) · [Contributing & Test Tiers](CONTRIBUTING.md)

We thank the open-source projects used in Zeta0:
- [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0): inherited and adapted code in the training library and deployment runtimes.
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): the native N1.7 action head used by the `gr00t_n1d7` backend.
- [LeRobot](https://github.com/huggingface/lerobot), [GR00T-WholeBodyControl](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl), and [HumanoidArena](https://github.com/William-wAng618/HumanoidArena): G1 capture, SONIC whole-body control, and simulation benchmark integrations.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
