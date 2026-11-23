# 真机数据最终方法微调标准

这份 G1/BrainCo 真机微调参考配方使用四卡训练
20,000 个 optimizer update。统一入口为
`scripts/train/predictive_world/finetune-real-sonic-final-4gpu-20k.sh`。

## 固定配方

- 当前一条实测 state41（q29 + 双手12），三路当前 RGB；动作 motion64 + hand12。
- FULL+ERT 双响应、目标合同 v5：teacher/student 响应各 0.5，外层权重 1；
  body 辅助权重 1、temporal 权重 0.25，保留普通动作监督。
- 从共同 joint-15000 基础权重初始化，每个新数据集独立微调；不默认接着旧任务权重训练。
- 4 个 DDP rank，每卡 batch 256、累积1，全局 batch 1024；20,000 更新。
- 每5,000更新保存，保留5k/10k/15k/20k四个恢复点和最终权重。
- 沿用真机 RTC15、30步动作@50Hz；当前采集快照、固定动作时间网格、
  按真实时间分别采样 motion/left/right 标签、使用训练记录动作范围。
- 接受经验证的30/50Hz存储网格；物理时间使用原始接收/控制周期时间，
  不把存储帧率当作传感器同步。v2 latest-available 合同保留其原始年龄上限，
  不套用v1相机配对的偏移限制。原生控制周期中，动作提交是观察之后的训练标签，
  不作为因果观察输入。
- 默认全部选定轨迹用于训练，不隐式划分验证集。数据排除同时作用于训练索引和
  归一化统计，保持原 episode_index，不修改原始压缩包。

## 调用

从项目根目录运行，路径和GPU编号按实际情况填写：

```bash
DATASET=/path/to/extracted_dataset \
OUTPUT_DIR=/path/to/unique_new_run \
TASK_ID=real/task_name \
INIT_JOINT=/path/to/joint-15000 \
SOURCE_CONFIG=/path/to/joint-15000.source-config.json \
ASSET_PATH_MAP=/path/to/local-assets.json \
PROCESSOR_MIGRATION=/path/to/verified-processor-migration.json \
PYTHON_BIN=/path/to/.venv-training/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train/predictive_world/finetune-real-sonic-final-4gpu-20k.sh
```

源码分发不包含站点 `.local.json`。上述source配置、路径映射和processor迁移必须绑定
实际初始化权重及运行环境；不要用占位文件替代。

可传入 `--audit-only`、`--prepare-only`、`--preflight-only` 检查对应阶段。
训练会依次进行数据准备、视觉特征缓存、三步四卡训练预检和正式微调。
需要排除轨迹时，将 `EXCLUDE_EPISODES` 设为要排除的原始索引，多个索引用逗号分隔。
请按本次数据集确定排除项。不同硬件需要调整批量时，记录实际使用的配方。

## 开源免私有权重训练（`--init-mode scratch` 与 `--init-mode gr00t_n1d7`）

外部用户无需 `joint-15000` 私有预训练权重或站点 `.local.json`，即可直接在 `SII-yizhi/zeta0-real-dataset` 或自采 G1 数据集上以 **v6/raw 联合目标** 训练并导出真机推理包：

- **30 Hz 扁平格式（`sonic_lerobot`，如 `place_chair`、`tidy_scene`、`clean_table_trash`、`blow_dry_hair`、`place_tape_on_table` 以及开源 G1 采集脚本直出数据）**：使用 `scripts/train/predictive_world/prepare_sonic_lerobot_task.py`，支持 `--whole-body-target` 监督全部 29 个全身实测关节。
- **50 Hz 分列格式（`real_sonic`，如 `place_doll`、`additional/flatten_box`）**：使用 `scripts/train/predictive_world/prepare_real_sonic_task.py`，配合 `--current-state-only --fixed-action-grid --independent-future-components --collector-snapshots --recorded-action-projection --real-sample-filter collector_snapshot_v1`（`place_doll` 按说明传 `--exclude-episode 80`）。

```bash
# 1. 从零初始化 Ψ₀ (zeta_native) + 公开 Qwen3-VL-2B
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode scratch --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model /path/to/Qwen3-VL-2B --output runs/place_chair_scratch \
  --current-state-only --fixed-action-grid --whole-body-target \
  --action-horizon 50 --gpus 1 --batch 64 --updates 20000

# 2. 从公开 NVIDIA GR00T N1.7-3B 初始化 (gr00t_n1d7，原生 40 步动作块，支持 30 Hz 与 50 Hz)
PYTHONPATH=src python scripts/train/predictive_world/prepare_sonic_lerobot_task.py \
  --init-mode gr00t_n1d7 --dataset /path/to/place_chair --task real/place_chair \
  --qwen-model /path/to/exported_gr00t_backbone --gr00t-model /path/to/GR00T-N1.7-3B \
  --output runs/place_chair_gr00t --current-state-only --fixed-action-grid --whole-body-target \
  --gpus 1 --batch 64 --updates 20000
```

也可通过 `finetune-real-sonic.sh` 直接设定 `INIT_MODE=scratch` 或 `INIT_MODE=gr00t_n1d7`（加 `GR00T_MODEL=...`）串起数据准备、特征缓存、3 步预检与训练。
