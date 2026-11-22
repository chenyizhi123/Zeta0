# CUDA contract acceptance — 2026-11-16

The bounded acceptance run passed on one **NVIDIA H200**: **3 passed, 0 skipped**
(8 CPU/nonselected cases deselected). The CUDA installation example also completed
eight optimizer updates, saved at update four, verified exact predictions after
reload, and resumed with optimizer/RNG state through update eight.

## Source and environment

- Reviewed source: `fb8d691bdb7103eeae3bbdefcf1d0701326b48b8` from the original source validation.
- Incoming 2,233-file source archive SHA256:
  `04cc070ac89c89b32aad67e9a00938964035ee92559f27700d1a489d3f0c68be`.
- The exported source was initialized as an independent Git snapshot,
  `39ddc78572d0ecddf2303a0b14d8f2e17f1fbc45`, for runtime provenance checks.
  This snapshot identity is not a different upstream release. No model or
  training implementation changed for this run.
- Native GR00T source pin: `3b49067943eb1ab2eb2203b93d461c4390c78da3`.
- Existing shared Python environment reused without package installation or
  modification: Python 3.10.18, Torch 2.7.0+cu126, torchvision 0.22.0,
  Transformers 4.57.3, Diffusers 0.35.1 and NumPy 1.26.4.
- Ubuntu 22.04.4, NVIDIA driver 595.58.03, Torch CUDA 12.6, H200 143,771 MiB.
  The container image was `pytorch-inspire-base:2.4.0-cuda12.4-cudnn9-runtime`;
  those image-label versions do **not** describe the selected shared Python.
- Offline Hugging Face access, task-local caches, bytecode writes disabled,
  one GPU, 8 GiB shared memory, a 15-minute platform runtime limit and a
  10-minute validation-command limit. No pretrained weights or demonstrations
  were downloaded or used; no robot commands were sent.

The initial attempt stopped before model computation because the minimal image
lacked `libGL.so.1`. Twelve graphics/GLib runtime libraries were copied from an
existing image with the same Ubuntu version into a separate validation directory,
verified by SHA256, and selected through the process library path. The shared
Python installation remained unchanged. Their CPU import check passed before
the successful GPU attempt. Public users should install the documented Ubuntu
packages from [installation prerequisites](../installation.md), rather than rely
on these private validation files. No system-library binaries enter this release.

## Observed results

| Check | Result |
| --- | --- |
| CUDA synthetic predictor/trainer/checkpoint example | 8 updates, resumed at 4; exact reload predictions |
| Synthetic MSE, seed 7 | 0.4241923094 → 0.3336610794 |
| BF16 world/joint optimizer chain | Passed; finite loss/gradients and updates |
| Native CUDA checkpoint/recomputation, normal and response calls | Passed; exact outputs, gradients, RNG and state keys |
| Native CUDA checkpoint/recomputation, response reader | Passed; exact outputs, gradients, RNG and state keys |
| Selected pytest cases | 3 passed, 0 skipped, 12.00 seconds |

Four warnings came from deprecated upstream Diffusers configuration attribute
access; all selected assertions passed. The job exited zero and reached
`SUCCEEDED`; its containers were released. Temporary preparation notebooks were
deleted. Existing user notebooks, environments, models and research files were
preserved. Terminal job records and private raw results remain available for
review without holding GPU containers.

[Reviewed machine-readable evidence](evidence/cuda-20261116.json) includes the
case names, actual versions, synthetic metrics and raw-result SHA256 identities.
Private filesystem paths and raw environment inventories are omitted. Re-run via
the [GPU quickstart instructions](../quickstart.md), using a fresh output path.

These are small synthetic and native-head contract checks. They establish CUDA
execution and the selected numerical/checkpoint boundaries. They do not establish
full-model learning quality, pretrained policy reproduction or physical robot
acceptance. The later archive, documentation and CPU test-supervision changes do not alter
the GPU-tested model, trainer, example, native source pin or selected CUDA tests.
