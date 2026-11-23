# 中文文档导航

[English](README.md) | **简体中文** | [返回项目首页](../README_zh.md)

从下面选择要完成的任务。项目概览见[中文首页](../README_zh.md)；详细技术指南目前主要为英文。

## 从这里开始

| 你的目标 | 推荐入口 |
| --- | --- |
| 了解框架定位、核心成绩与分层架构 | [中文首页](../README_zh.md)、[路线图](../ROADMAP.md) |
| 复用已有 Python 环境与本地资产 | [环境检查](framework.md#start-with-your-existing-environment) |
| 首次安装、Docker 镜像与公开数据集 | [安装指南](installation.md)、[Docker 环境](../docker/README.md)、[公开依赖与资产](public_assets.md) |
| 无需机器人、数据集或模型权重，30 秒跑通全算法演示并出图 | [最小运行示例](quickstart.md) |
| 3D 打印腕部相机支架（RealSense D415） | [腕部相机支架与打印指南](../hardware/README.md) |
| 使用统一命令入口（`zeta`） | [框架 CLI](framework.md) |

已有环境时，在仓库根目录运行以下只读检查，将路径替换为已有环境的 Python：

```bash
PYTHONPATH=src python -m psi.cli doctor --profile train --python /path/to/existing/python
```

训练与 G1 采集分别使用 Python 3.10 和 Python 3.12 环境，具体依赖按安装指南选择。

## 三条工作流

| 工作流 | 操作入口 | 配套说明 |
| --- | --- | --- |
| 数据采集与检查 | [G1 数据采集](../real/g1/README.md#collection)、[数据审计](framework.md#inspect-captured-data) | [真实数据适配](real_data_final_method.md)、[数据准备与复现](reproducibility.md) |
| 策略训练 | [后端选择与安装](backends.md)、[训练与预检](backends.md#train-and-preflight) | [Psi0-derived 配置](../configs/psi0/README.md)、[GR00T N1.7 配置](../configs/gr00t_n1d7/README.md) |
| 真机部署与诊断 | [G1 部署](../real/g1/README.md#deployment)、[配置新任务](framework.md#add-a-task-without-editing-python) | [运行时兼容性](../real/g1/README.md#runtime-compatibility)、[离线日志诊断](framework.md#diagnose-a-recorded-session-offline) |

训练数据、任务权重和硬件资产按[外部资产要求](artifacts.md)准备。
支持的模型与运行时见[兼容矩阵](workflows.md)。

## 开发与维护

| 主题 | 文档 |
| --- | --- |
| 安装检查与测试范围 | [CPU 检查](../CONTRIBUTING.md#cpu-checks)、[测试分层](../CONTRIBUTING.md#test-tiers) |
| 贡献代码 | [贡献指南](../CONTRIBUTING.md)、[源码职责](maintenance/ARCHITECTURE.md) |
| 模型与数据接口约定 | [模型方法](predictive_world.md)、[研究与兼容性约定](maintenance/RESEARCH_CONTRACTS.md) |
| 第三方依赖与许可证 | [第三方说明](../THIRD_PARTY_NOTICES.md) |
| 发布审查与已验证范围 | [G1 源码审查](maintenance/G1_INTEGRATION_REVIEW.md)、[发布检查](maintenance/RELEASE_CHECKLIST.md) |
| 安全问题反馈 | [安全说明](../SECURITY.md) |

更多专题与历史研究背景见[完整文档索引](README.md)。
