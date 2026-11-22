# Zeta0

**Demonstration collection, policy training and real-robot deployment for humanoid VLA and WAM.**

English | [简体中文](README_zh.md)

Zeta0 connects Unitree G1 demonstration collection, model training and
policy execution. The training core supports Psi0-derived and native GR00T N1.7
policies. The G1 tools integrate SONIC whole-body control and BrainCo hands.

![Zeta-0 policy: autonomous G1 task, scene-and-body prediction, and additional real-world tasks](docs/assets/zeta0-overview.png)

*Zeta-0, the current model developed in the framework: scene-and-body foresight
conditions whole-body actions. See the [model design](docs/predictive_world.md).*

## Released source

**2026-11-22 · v0.1.0.** This release includes G1 teleoperation and LeRobot v3
collection, data preparation, both training backends, checkpoint export and
matching G1 deployment runtimes. It also includes setup guides and offline tests.

To smooth the handoff between action chunks on G1, deployment uses RTC
(Real-Time Chunking) to predict the next chunk while the robot executes the
current one.

## Highlights

- **Collect and prepare demonstrations.** Record G1 data in LeRobot v3, inspect
  state/action columns and timing, and prepare task-specific training inputs.
- **Train with either model host.** Psi0-derived and native GR00T N1.7 hosts share
  prediction, objectives, checkpointing and resume tools.
- **Deploy a matching policy on G1.** Export a completed joint checkpoint or
  use an existing task export with its corresponding runtime.
- **Configure tasks and diagnose sessions.** Add a task through configuration,
  check model packages and inspect recorded execution logs with `zeta`.

## Quick start

```bash
git clone https://github.com/chenyizhi123/Zeta0.git
cd Zeta0
```

Follow [installation](docs/installation.md) for your workflow. If you already have
an environment, [check and reuse it](docs/framework.md#start-with-your-existing-environment).

For a first run, the [quickstart](docs/quickstart.md) trains a small predictor
on synthetic data, saves a checkpoint and resumes training. With its CPU
environment ready:

```bash
make demo PYTHON=.venv-check/bin/python
```

Results are saved under `runs/quickstart-*`; `summary.json` reports the reload and
resume checks. Models, datasets and robot hardware are unnecessary for this example.

## Collection → training → deployment

| Goal | Guide |
| --- | --- |
| Collect G1 demonstrations | [Capture setup and recording](real/g1/README.md#collection), [data inspection](docs/framework.md#inspect-captured-data) |
| Prepare data and train | [Backend setup](docs/backends.md), [preparation and training](docs/reproducibility.md) |
| Run a policy on G1 | [Research service and robot client](real/README.md), [G1 task runtimes](real/g1/README.md#deployment) |

Training uses Python 3.10 and G1 capture uses Python 3.12. The documented
G1 data preparation and fine-tuning recipe uses the Psi0-derived host. Native
GR00T N1.7 is available as a separate training backend with its own input
requirements. Historical G1 task runtimes require matching task weights and
observation/action contracts. See the [workflow guide](docs/workflows.md) and
[asset catalog](docs/public_assets.md).

## Policy implementations

Zeta-0 predicts scene-and-body futures from current observations and uses them to
condition actions. Two implementations share the same training core:

| Model host | Configuration | Guide |
| --- | --- | --- |
| Psi0-derived | `model.host_architecture=zeta_native` | [Psi0-derived recipe](configs/psi0/README.md) |
| Native GR00T N1.7 | `model.host_architecture=gr00t_n1d7` | [GR00T recipe](configs/gr00t_n1d7/README.md) |

Task-trained weights and datasets are [external assets](docs/artifacts.md).

## Documentation

[Documentation index](docs/README.md) · [CLI guide](docs/framework.md) ·
[Contributing and tests](CONTRIBUTING.md)

## Acknowledgments and licenses

We thank the open-source projects used in Zeta0:

- [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0): inherited and
  adapted code in the training library and deployment runtimes.
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): the native N1.7 action
  head used by the `gr00t_n1d7` backend.
- [LeRobot](https://github.com/huggingface/lerobot) and
  [GR00T-WholeBodyControl](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl):
  G1 data collection and SONIC control integrations.

See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md).
