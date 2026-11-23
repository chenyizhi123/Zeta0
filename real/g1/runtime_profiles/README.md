# G1 Runtime Profiles

Zeta0 v0.2+ uses a **single unified G1 runtime** (`unified`) backed directly by
the maintained repository `src/psi` implementation (`ResearchInferenceRuntime` +
Layered VLA Host Registry). It supports:

- **`zeta_native`** (Ψ₀-derived, both v6/raw and historical whitened exports; 30Hz/50 or 50Hz/30)
- **`gr00t_n1d7`** (native GR00T N1.7 v6/raw exports; 50Hz/40 or 30Hz/40)
- Future layered hosts (such as `pi05`) without duplicating the transport or RTC stack.

Set `"runtime_profile": "unified"` in your task's `model_paths.json` (or
`ZETA0_POLICY_PROFILE=unified`).

## Archived legacy v5 task snapshots

In v0.1, five historical task runtimes (`blow_dry_hair_runtime_20261029`,
`clean_table_runtime_local_v1`, `place_chair_runtime_20261029`,
`tidy_scene_runtime_local_v1`, `zeta0_deploy_offline_kit_20261024`) were stored
as five duplicated 46-file copies (~60,356 lines, 35/46 files identical; see
[`COMPARISON.json`](COMPARISON.json)).

Those exact byte-for-byte historical snapshots are preserved on the archival
branch `archive/legacy-v5-runtimes` (commit `6965053`) and indexed with SHA-256
digests in [`LEGACY_ARCHIVE.json`](LEGACY_ARCHIVE.json). If you need to replay a
legacy v5 weight bundle on G1, restore them locally with:

```bash
git checkout archive/legacy-v5-runtimes -- real/g1/runtime_profiles
# Or extract legacy-g1-v5-runtimes-v0.1.tar.gz into real/g1/.local/zeta0/
```
