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

## Capture-to-training boundary

Use the native `sonic_g1` adapter for official LeRobot v3 recordings. Historical
v2.1 recipes use a different loader. Before training, check state/action columns,
normalization, timestamps, episode quality and splits with
[data inspection](framework.md#inspect-captured-data), then follow
[real-data preparation](real_data_final_method.md).
