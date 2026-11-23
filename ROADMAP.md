# Zeta0 Roadmap

Zeta0 focuses on turning **hierarchical / dual-system VLA policies** (System 2 VLM + System 1 flow-matching action expert) into **World Action Models (WAM)** with joint scene-and-body foresight for whole-body humanoid control.

## Completed (v0.2.0)

- [x] Shared scene-and-body foresight core (`E/EMA`, `D`, `H`, `F` prior/world, Dual-Response Transfer `L_resp`, causal `PredictiveWorldPolicy`).
- [x] Layered VLA Host Registry (`zeta_native` Ψ₀ + `gr00t_n1d7` GR00T N1.7) unified across training, export, and G1 deployment.
- [x] Public `zeta0` Python SDK facade with backward-compatible `psi` checkpoint contracts.
- [x] `zeta` CLI end-to-end operations (`doctor`, `init`, `data pull/inspect`, `bundle inspect`, `logs summarize`, `visualize`, `workflow`).
- [x] Full-algorithm CPU quickstart with automatic foresight preview plotting + `zeta visualize` (PNG/HTML + optional Rerun 3D).
- [x] HumanoidArena closed-loop evaluation bridge, hardware-free G1 mock client, Dockerfiles, and slimmed hardware asset distribution.

## Next (v0.3.x)

- [ ] **π₀.₅ (`pi05`) Layered Host**: Implement `src/psi/models/predictive_world/hosts/pi05.py` using the PaliGemma KV-cache + SigLIP visual patch + Gemma action-expert contracts prepared in v0.2.0.
- [ ] **Public HuggingFace Model Zoo**: Publish reviewed HumanoidArena 7-task checkpoints (`ζ₀-Ψ₀` and `ζ₀-GR00T-N1.7`) with `PredictiveWorldPolicy.from_pretrained(...)` loader support.
- [ ] **TensorRT / `torch.compile` Deployment Profile**: Push closed-loop RTX 4090 inference latency below 120 ms with verified numerical equivalence tests.
- [ ] **Public Real Dataset Card & Sample Subset**: Finalize public visibility and LeRobot v3 viewer metadata on `SII-yizhi/zeta0-real-dataset`.
- [ ] **STEP CAD Sources**: Publish parametric `.STEP` CAD files for the wrist camera mounts to support community remixes (D405 / D435i / Orbbec).
