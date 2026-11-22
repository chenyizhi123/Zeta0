# Installation

Start by checking your existing environment with the
[framework CLI](framework.md#start-with-your-existing-environment).
For a fresh setup, choose an environment from the table below.

Clone the repository to use data preparation, training scripts and G1 tools.

```bash
git clone https://github.com/chenyizhi123/Zeta0.git
cd Zeta0
```

The Python package is distributed as `zeta0-humanoid` and installs the `zeta`
command. Imports retain `psi` for compatibility. The wheel contains the library.
Training scripts and hardware tools run from the checkout.

| Purpose | Platform/environment | Guide |
| --- | --- | --- |
| Offline CPU checks | Python 3.10; isolated `.venv-check` | [CPU checks](../CONTRIBUTING.md#cpu-checks) |
| Train/serve Psi0-derived Zeta | Linux CUDA, Python 3.10, uv 0.8.20, `psi`/`serve`/`robot` groups | [Backend installation](backends.md#install) |
| Train/serve native GR00T N1.7 | Separate Python 3.10 environment, additional `gr00t` group and exact source pin | [Backend installation](backends.md#install) |
| Official LeRobot G1 capture | Linux x86_64, Python 3.12, separate `.venv_lerobot` | [G1 setup](../real/g1/README.md#install-and-configure) |
| SONIC and real-model inference | Original control virtual environments, native controller, CUDA/TensorRT and matching model assets | [G1 setup](../real/g1/README.md#install-and-configure) |

On Ubuntu 22.04, native video and OpenCV tests require these system libraries:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
```

Use the [CPU setup](../CONTRIBUTING.md#cpu-checks) for offline checks and the
[backend setup](backends.md#install) for CUDA training. The root `uv.lock` covers
the library and training dependencies. G1 capture uses its own installer and
`real/g1/scripts/requirements-lerobot.txt`; keep it in a separate environment.

See [the workflow support matrix](workflows.md), [external artifact requirements](artifacts.md)
and [contributing test tiers](../CONTRIBUTING.md#test-tiers).
