# Changelog

All notable changes to Zeta0 are documented in this file.

## [0.2.0] - 2026-11-23

### Architecture & Layered VLA Hosts
- Introduced the **`zeta0` top-level public Python SDK namespace** (`src/zeta0/`) alongside backward-compatible `psi` imports.
- Added a unified **Layered VLA Host Registry** (`src/psi/models/predictive_world/hosts/`) supporting `zeta_native` (Ψ₀) and `gr00t_n1d7` (GR00T N1.7) across training, export, and deployment, with explicit `pi05` (π₀.₅) contract scaffolding (`kv_cache` support in `ObservationFeatures`, parameterized `visual_dim` in `FutureTrajectoryEncoder` / `WorldReadout`).
- Unified **`ResearchInferenceRuntime`** and **`export_real_sonic_research_policy`** so both `zeta_native` and `gr00t_n1d7` v6/raw checkpoints export and load cleanly through the same G1 deployment runtime.
- Archived the five duplicated legacy v5 runtime snapshots (60,356 lines across 231 files) to branch `archive/legacy-v5-runtimes` and `LEGACY_ARCHIVE.json`, introducing a lightweight `unified` G1 runtime profile in `real/g1/runtime_profiles/`.

### Developer Experience, Visualization & Tooling
- Upgraded `scripts/examples/quickstart.py` (`make demo`) to execute the complete Zeta-0 algorithm (future encoder `E/EMA`, readout `D`, shared `prior/world` predictor `F`, independent teacher/student `T/G` with dual-response transfer `L_resp`, causal deploy-only `PredictiveWorldPolicy`, checkpoint reload/resume, and automatic `foresight_preview.png` generation).
- Added **`zeta data pull`** (wired to `SII-yizhi/zeta0-real-dataset` with per-task ZIP selection), **`zeta init`** (automated config scaffolding without manual `REQUIRED_*` editing), and **`zeta visualize`** (offline PNG/HTML & optional Rerun 3D foresight + `RuntimeAudit` RTC latency dashboards).
- Added standalone no-private-weights real-robot training (`--init-mode scratch` and `--init-mode gr00t_n1d7` across 30 Hz and 50 Hz G1 tasks), hardware-free **mock G1 RTC client** (`scripts/deploy/mock_g1_client.py`), **inference latency benchmark** (`scripts/benchmark_inference.py`), and official **`docker/`** images.

### Documentation & Repository Hygiene
- Updated bilingual `README.md` / `README_zh.md` with project page, paper PDF, real-robot G1 gallery, core results, datasets, architecture overview, and BibTeX; updated `CITATION.cff` with full paper citation.
- Moved oversized STL meshes and URDF camera FOV preview videos out of Git history into the verified `hardware-wrist-camera-v1` asset manifest (`hardware/README.md`, `hardware/assets_manifest.json`), restoring full `make check-release` compliance, and removed tracked `.DS_Store` files.

## [0.1.0] - 2026-11-22

- Initial verified open-source release with G1 teleoperation/LeRobot v3 capture, Ψ₀-derived and native GR00T N1.7 training backends, RTC deployment tools, and offline CPU/G1/GR00T regression suites.
