# ζ₀ (Zeta0)

**面向人形机器人控制的联合场景与身体预见世界动作模型框架**

[![项目主页](https://img.shields.io/badge/项目主页-ζ₀-0F766E?style=flat-square)](https://chenyizhi123.github.io/zeta0-project/)
[![论文 PDF](https://img.shields.io/badge/论文-PDF_(18页)-B91C1C?style=flat-square)](https://chenyizhi123.github.io/zeta0-project/assets/Zeta0-Humanoid-Full.pdf)
[![仿真数据集](https://img.shields.io/badge/HF_数据集-HumanoidArena_SONIC40-F59E0B?style=flat-square)](https://huggingface.co/datasets/SII-yizhi/HumanoidArenaV3.1_SONIC40)
[![真机数据集](https://img.shields.io/badge/HF_数据集-Zeta0_Real_G1-F59E0B?style=flat-square)](https://huggingface.co/datasets/SII-yizhi/zeta0-real-dataset)
[![许可证](https://img.shields.io/badge/License-Apache_2.0-2563EB?style=flat-square)](LICENSE)

[English](README.md) | 简体中文 · [项目主页](https://chenyizhi123.github.io/zeta0-project/) · [论文 PDF](https://chenyizhi123.github.io/zeta0-project/assets/Zeta0-Humanoid-Full.pdf) · [路线图](ROADMAP.md) · [更新日志](CHANGELOG.md)

**Zeta0** 是一个将分层/双系统视觉-语言-动作策略（VLA）转化为**世界动作模型（WAM）**的开源框架：它在共享潜表示中联合预见未来视觉特征与实测全身关节轨迹，并通过**双响应迁移（Dual-Response Transfer）**在动作空间直接监督因果预测器。代码库完整打通 Unitree G1 + SONIC + BrainCo 示范采集、**Ψ₀** 与 **GR00T N1.7** 双分层底座联合训练与统一导出部署（已预留 **π₀.₅** 分层接入契约）、HumanoidArena 闭环评测桥、离线预见可视化以及 **RTC（Real-Time Chunking）** 实时执行。

![Zeta-0 模型：G1 自主任务、场景与身体未来预测，以及其他真机任务](docs/assets/zeta0-overview.png)

*Zeta-0 从因果观测预见未来 +0.2 / +0.4 / +0.6 s 的场景与身体变化，并以此引导全身动作。详细方法见[模型设计说明](docs/predictive_world.md)。*

---

## 核心实验结果

| 基准 / 指标 | 底座基线 | **ζ₀（本文）** | 提升 | 说明 |
| --- | :---: | :---: | :---: | --- |
| **HumanoidArena（7 任务均值，Ψ₀）** | 44.8% | **63.6%** | **+18.8 pp** | 每设置 420 局 · `zeta_native` |
| **HumanoidArena（7 任务均值，GR00T N1.7）** | 70.5% | **79.5%** | **+9.0 pp** | 超过所有对比七任务联合策略 · `gr00t_n1d7` |
| **Unitree G1 真机（6 任务，Ψ₀）** | 61.7% | **78.3%** | **+16.7 pp** | 300 次物理实测 · 多阶段任务 +26.7 pp |
| **Unitree G1 真机（6 任务，GR00T N1.7）** | 75.0% | **85.0%** | **+10.0 pp** | SONIC 全身控制 + BrainCo 灵巧手 + RTC |
| **29 关节未来身体预测（+0.6 s RMSE）** | 9.36°（保持） | **4.53°** | **−51.6%** | 526 个窗口胜率 83.5%（腿部 5.23° vs 12.54°） |
| **闭环推理延迟（单张 RTX 4090）** | — | **212 ms** | **6.3 GiB** | 每个动作块仅需一次确定性未来预测 |

---

## 分层架构（`System 2 → 1.5 → 1 → 0`）

Zeta0 作为因果 **System 1.5 预见桥** 嵌入在冻结 System 2 视觉语言模型与 System 1 流匹配动作专家之间，并通过分层 Host 注册表（`zeta0.hosts` / `psi.models.predictive_world.hosts`）贯通训练、导出与真机部署三端：

| 层级 | 模块职责 | 当前与规划实现 |
| --- | --- | --- |
| **System 2（VLM）** | 当前多模态上下文与视觉 patch 底座 | 冻结 Qwen3-VL（`zeta_native`）、导出 GR00T N1.7 VLM（`gr00t_n1d7`）、PaliGemma + KV-Cache（`pi05`，规划中） |
| **System 1.5（ζ₀ 核心）** | 共享潜表示与动作空间响应迁移 | 仅训练态 `E_φ / E_φ̄` + `WorldReadout D` · 部署态因果 `H` + `F_θ(prior/world)` · 双响应迁移 `L_resp` |
| **System 1（动作专家）** | 流匹配动作块生成 + RTC 前缀衔接 | `TemporalNativeActionExpert`（`zeta_native`）、原生 `Gr00tN1d7ActionHead`（`gr00t_n1d7`）、Gemma 动作专家（`pi05`，规划中） |
| **System 0（全身控制）** | Unitree G1 实机执行 | 统一 G1 运行时（`unified`）· RTC 实时分块 · GEAR-SONIC · BrainCo Revo2 灵巧手 |

> 部署态通过 `PredictiveWorldPolicy` 在结构上彻底剥离未来编码器 `E_φ`、EMA 编码器、读出头 `D`、教师 `T_ψ` 与世界模式动作标签。

---

## 四大真机开源主线（Unitree G1 + SONIC + BrainCo）

本次开源聚焦于**真机人形机器人从硬件到部署的完整可复现闭环**：

| 开源主线 | 核心内容 | 文档与入口 |
| --- | --- | --- |
| **1. 硬件打印方案（3D 可打印）** | RealSense D415 腕部支架（论文版手背 `+30°` 与拓展版手心侧 `−20°..+20°`，推荐优先测试 `0°`）、打印材料/壁厚/M3 螺丝 BOM、URDF 预览与 SHA-256 清单 | [`hardware/README.md`](hardware/README.md) · [`hardware/assets_manifest.json`](hardware/assets_manifest.json) |
| **2. 真机采集方案** | PICO / XRoboToolkit 遥操作 + GEAR-SONIC 全身控制 + BrainCo Revo2 灵巧手 + 三路 D415 直出 LeRobot v3（`state41` / `motion76`），配合 `zeta data inspect` 质量审计 | [`real/g1/README.md`](real/g1/README.md#collection) · [`docs/framework.md`](docs/framework.md#inspect-captured-data) |
| **3. 真机训练方案（免私有权重）** | 支持从公开 **Qwen3-VL / Cosmos-Reason2-2B**（`--init-mode scratch`）或 **NVIDIA GR00T N1.7-3B**（`--init-mode gr00t_n1d7`，兼容 30 Hz 与 50 Hz）直接训练 v6/raw 真机策略 | [`scripts/train/predictive_world/prepare_sonic_lerobot_task.py`](scripts/train/predictive_world/prepare_sonic_lerobot_task.py) · [`docs/real_data_final_method.md`](docs/real_data_final_method.md) |
| **4. 真机部署方案** | 统一 G1 运行时（`unified`）与 `real_sonic_server`，原生支持 `zeta_native`（`50/30`、`30/50`）与 `gr00t_n1d7`（`30/40`、`50/40`）RTC 在线推理，并提供无真机 `mock_g1_client.py` 与 `zeta visualize log` | [`real/g1/runtime_profiles/README.md`](real/g1/runtime_profiles/README.md) · [`real/g1/README.md`](real/g1/README.md#deployment) |

### 真机数据免私有权重训练示例

```bash
# 方案 A：使用公开 Qwen3-VL-2B 从零训练 Zeta-0（Ψ₀ / zeta_native，30 Hz 真机数据）
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode scratch --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model /path/to/Qwen3-VL-2B --output runs/place_chair_scratch \
  --current-state-only --fixed-action-grid --whole-body-target --action-horizon 50 \
  --gpus 1 --batch 64 --updates 20000

# 方案 B：接入公开 NVIDIA GR00T N1.7-3B（30 Hz / 40 步动作块）训练并直出部署 G1
PYTHONPATH=src python scripts/train/gr00t_n1d7/export_backbone.py \
  --base /path/to/GR00T-N1.7-3B --cosmos-processor /path/to/Cosmos-Reason2-2B --output assets/gr00t_n1d7_backbone
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode gr00t_n1d7 --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model assets/gr00t_n1d7_backbone --gr00t-model /path/to/GR00T-N1.7-3B \
  --output runs/place_chair_gr00t --current-state-only --fixed-action-grid --whole-body-target \
  --gpus 1 --batch 64 --updates 20000

# 随后执行特征缓存 → 训练 → 导出部署包 → 启动推理服务（GR00T N1.7 加 --gr00t-model）
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

## 已开源内容与公开数据集

**2026-11-23 · v0.2.0。** 开放 `zeta0` 顶层 Python SDK 门面、贯通训练/导出/真机部署的分层 VLA Host 注册表（`zeta_native` + `gr00t_n1d7`，已为 `pi05` 预留 `kv_cache` 与动态 `visual_dim` 契约）、免私有权重真机训练入口（`--init-mode scratch` 与 `--init-mode gr00t_n1d7`，全面支持 30 Hz 与 50 Hz 真机任务）、完整算法 CPU 最小演示与可视化报告、`zeta visualize` 预见与 RTC 延迟看板、HumanoidArena 评测桥、无真机模拟 G1 客户端与官方 Dockerfile。

| 资产 | 获取地址 | 格式与规格 |
| --- | --- | --- |
| **真机 G1 数据集（6+1 任务）** | [`SII-yizhi/zeta0-real-dataset`](https://huggingface.co/datasets/SII-yizhi/zeta0-real-dataset) | Unitree G1 + BrainCo + SONIC · 739 条轨迹 · 872,409 帧（约 24.4 GB，按任务 ZIP + `SHA256SUMS`）· 5 项 30 Hz 扁平（`sonic_lerobot`）+ 2 项 50 Hz 分列（`real_sonic`） |
| **仿真数据集（7 任务）** | [`SII-yizhi/HumanoidArenaV3.1_SONIC40`](https://huggingface.co/datasets/SII-yizhi/HumanoidArenaV3.1_SONIC40) | 已公开 · LeRobot · 700 条轨迹 · 546,460 帧 · 50 Hz · `state64` / `semantic40` |
| **基础模型与硬件 SDK** | [公开资产目录](docs/public_assets.md) | Qwen3-VL / Ψ₀ 底座、NVIDIA GR00T N1.7-3B、GEAR-SONIC、BrainCo SDK、XRoboToolkit |
| **任务微调权重** | [外部资产要求](docs/artifacts.md) | 外部独立管理（不包含在本源码仓库内） |

---

## 快速开始（CPU 30 秒完整跑通，无需权重或真机）

```bash
git clone https://github.com/chenyizhi123/Zeta0.git
cd Zeta0

# 1. 创建轻量 CPU 环境（Python 3.10）
uv venv --python 3.10 .venv-check
uv pip install --python .venv-check/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-check/bin/python --group cpu-test

# 2. 跑通完整 Zeta-0 联合训练（E/EMA + D + F prior/world + T/G 双响应迁移 + 因果部署 + 自动出图）
make demo PYTHON=.venv-check/bin/python
```

示例会在 `runs/quickstart-*` 下输出 `summary.json`、`predictor.safetensors` 和自动生成的 **`foresight_preview.png`**（含训练曲线、第 4 步断点续训验证、以及 `D(ẑ_prior)` 未来身体轨迹预测对比）。

### Python SDK 入口（`import zeta0`）

```python
import zeta0
from zeta0.core import (
    FutureTrajectoryEncoder,
    WorldReadout,
    TemporalLatentTransition,
    JointObjective,
    PredictiveWorldPolicy,
)
from zeta0.hosts import list_hosts

print(zeta0.__version__, list_hosts())
# 0.2.0 ('gr00t_n1d7', 'pi05', 'zeta_native')
```

---

## `zeta` 统一命令行工作流

```bash
# 检查已有 Python 环境与已注册的分层 VLA Host
PYTHONPATH=src python -m psi.cli doctor --profile train --python /path/to/python
PYTHONPATH=src python -m psi.cli hosts

# 拉取仿真或真机数据集（默认只打印计划；加 --execute 执行下载）
zeta data pull humanoidarena --tasks HOI_pp_box --output data/humanoidarena
zeta data pull real --output data/zeta0-real

# 一键生成 v6/raw 训练配置（免手改 90+ 行 REQUIRED_* JSON）
zeta init --host gr00t_n1d7 --qwen-model /path/to/backbone --gr00t-model /path/to/GR00T-N1.7-3B --output configs/runs/pp_box.json

# 可视化未来身体预见曲线或 G1 RuntimeAudit RTC 延迟时间线
zeta visualize rollout runs/quickstart-*/summary.json --output runs/foresight.png
zeta visualize log runs/mock_g1_trace.jsonl --output runs/rtc_dashboard.png

# 无需真机 G1 即可干跑验证 RTC 协议与审计日志
PYTHONPATH=src python scripts/deploy/mock_g1_client.py --in-process --cycles 12 --output runs/mock_g1_trace.jsonl

# 端侧推理延迟基准（支持 CPU/CUDA 与 --compile）
PYTHONPATH=src python scripts/benchmark_inference.py --preset tiny-cpu --device cpu
```

---

## 全流程导航：采集 → 训练 → 评测 → 部署

| 目标 | 指南与入口 |
| --- | --- |
| **安装与 Docker** | [安装指南](docs/installation.md) · [Docker 镜像](docker/README.md) · [复用已有环境](docs/framework.md#start-with-your-existing-environment) |
| **采集 G1 示范** | [G1 采集与遥操作](real/g1/README.md#collection) · [数据质量审计](docs/framework.md#inspect-captured-data) |
| **训练策略（Ψ₀ / GR00T N1.7）** | [双后端工作流](docs/backends.md) · [复现步骤](docs/reproducibility.md) · [Ψ₀ 配置](configs/psi0/README.md) · [GR00T 配置](configs/gr00t_n1d7/README.md) |
| **HumanoidArena 仿真评测** | [HumanoidArena 7 任务评测桥](benchmarks/humanoidarena/README.md) |
| **Unitree G1 真机部署** | [统一 G1 运行时](real/g1/runtime_profiles/README.md) · [研究推理服务](real/README.md) · [G1 部署流程](real/g1/README.md#deployment) |

| 模型底座 | `model.host_architecture` | 状态 | 默认步长 / 求解步数 | 配置模板 |
| --- | --- | :---: | :---: | --- |
| **Ψ₀-derived** | `zeta_native` | 已支持（训练 + 导出 + G1 统一部署） | 50 步 / 10 步 Euler | [`configs/psi0/joint.template.json`](configs/psi0/joint.template.json) |
| **原生 GR00T N1.7** | `gr00t_n1d7` | 已支持（训练 + 导出 + G1 统一部署） | 40 步 / 4 步 Euler | [`configs/gr00t_n1d7/joint.template.json`](configs/gr00t_n1d7/joint.template.json) |
| **π₀.₅（PaliGemma + Gemma）** | `pi05` | 契约已预留（支持 `kv_cache`） | 50 步 / 10 步 Euler | [`src/psi/models/predictive_world/hosts/pi05.py`](src/psi/models/predictive_world/hosts/pi05.py) |

---

## 开源腕部相机支架（Unitree G1 + RealSense D415）

[`hardware/`](hardware/README.md) 提供 D415 腕部安装支架的打印参数、螺丝规范、URDF 预览与资产清单。为保持源码仓轻量，大体积 `.stl` 模型与仿真视场视频统一打包在发布资产 `hardware-wrist-camera-v1.tar.gz` 中（完整 SHA-256 见 [`hardware/assets_manifest.json`](hardware/assets_manifest.json)）。

| 支架 | 相机位置 | 角度 | URDF 预览 |
| --- | --- | --- | --- |
| 论文版支架 | 手背 | `+30°` | [30°](hardware/wrist-camera-paper/urdf_30deg.png) |
| 拓展版支架 *（推荐优先测试 **0°**）* | 手心一侧 | `−20°`、`−10°`、**`0°`**、`+10°`、`+20°` | [−20°](hardware/wrist-camera-expansion/urdf_neg20deg.png) · [−10°](hardware/wrist-camera-expansion/urdf_neg10deg.png) · [**0°**](hardware/wrist-camera-expansion/urdf_0deg.png) · [10°](hardware/wrist-camera-expansion/urdf_10deg.png) · [20°](hardware/wrist-camera-expansion/urdf_20deg.png) |

---

## 引用

如果 Zeta0 对你的研究或机器人系统有帮助，欢迎引用论文：

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

## 文档、致谢与许可

[中文文档导航](docs/README_zh.md) · [英文文档索引](docs/README.md) · [命令行指南](docs/framework.md) · [贡献与测试](CONTRIBUTING.md)

感谢 Zeta0 使用的开源项目：
- [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0)：训练代码库和部署运行时中继承与修改的代码来源。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：`gr00t_n1d7` 后端使用的原生 N1.7 动作头。
- [LeRobot](https://github.com/huggingface/lerobot)、[GR00T-WholeBodyControl](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl) 与 [HumanoidArena](https://github.com/William-wAng618/HumanoidArena)：G1 数据采集、SONIC 全身控制与仿真评测集成。

许可与归属说明见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。
