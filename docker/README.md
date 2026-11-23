# Official Zeta0 Docker Environments

These Dockerfiles isolate the three Python/CUDA stacks required across Zeta0's
workflows so users do not need to manage conflicting local virtual environments.

| Image | Target workflow | Python | Base |
| --- | --- | --- | --- |
| [`train-psi0.Dockerfile`](train-psi0.Dockerfile) | Psi0-derived (`zeta_native`) training, caching, export & HTTP serving | 3.10 | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` |
| [`train-gr00t.Dockerfile`](train-gr00t.Dockerfile) | Native GR00T N1.7 (`gr00t_n1d7`) training, export & serving | 3.10 | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` |
| [`g1-capture.Dockerfile`](g1-capture.Dockerfile) | Unitree G1 LeRobot v3 capture & diagnostics | 3.12 | `ubuntu:22.04` |

## Build examples

Run from the repository root:

```bash
# Psi0-derived training & serving image
docker build -f docker/train-psi0.Dockerfile -t zeta0:train-psi0 .

# GR00T N1.7 training & serving image
docker build -f docker/train-gr00t.Dockerfile -t zeta0:train-gr00t .

# G1 LeRobot v3 capture image
docker build -f docker/g1-capture.Dockerfile -t zeta0:g1-capture .
```

## Run with mounted datasets and models

```bash
docker run --gpus all --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/models:/workspace/models" \
  -v "$PWD/runs:/workspace/runs" \
  zeta0:train-gr00t \
  zeta doctor --profile train
```

For GR00T N1.7 inside the container, acquire the pinned upstream source once:

```bash
python scripts/fetch_sources.py Isaac-GR00T --apply
python scripts/check_source_pins.py Isaac-GR00T
```
