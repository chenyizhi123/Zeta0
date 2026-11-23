# G1 collection and deployment

Teleoperation, data collection and policy deployment for Unitree G1 with SONIC
and BrainCo hands. Run commands from `real/g1` unless stated otherwise.

## Install and configure

Use a Linux x86_64 workstation. Capture runs in Python 3.12; training uses
Python 3.10. SONIC control and policy inference each use their own environment.

From the repository root:

```bash
python scripts/fetch_sources.py GR00T-WholeBodyControl --apply
python scripts/setup_g1_dependencies.py --apply
cd real/g1
python configure_local_paths.py
```

Install the BrainCo/Unitree libraries, PICO service and controller models listed
in the [asset catalog](../../docs/public_assets.md).

Edit `config/local.json` with your equipment settings. Set `robot_interface`
to the workstation's wired interface, `robot_host` to the robot computer, and
`body_dds_peer` to the address of the Unitree body controller serving
`rt/lowstate`. The robot computer and body controller can use different
addresses. For an RTL8125 workstation using the optional network helper, pass
its PCI address and the workstation's static IPv4 address with prefix to
`scripts/configure_host_network.sh`.

Verify wrist camera identity from live images: sensor and USB serials can
differ. Configure the ROS head bridge's `~serial` to match
`head_camera_serial` and `~endpoint` to the bind address corresponding to
`camera_sources.ego_view`. The permission helper takes
`--expected-usb-serial`. Install camera scripts and their dependencies under
`robot_camera_root` on the robot computer. The optional wrist watchdog
installer uses the local unit specified by `--service-unit`; configure that
unit with the same root and the installed `wrist-config.json`.

Create `config/hand-hardware.json`, `config/head-startup-pose.json` and
`config/head-user-reference.json` using your robot's calibration. The matching
`.example.json` files describe the required fields. Set `head_servo_device`
to the verified persistent device path. On the robot, configure its head
services with `ZETA0_HEAD_SERVO_DEVICE` and the absolute
`ZETA0_HEAD_SETUP_ROOT` matching `robot_head_root`. If the servo calibration
file is outside the standard sibling service tree, set
`ZETA0_HEAD_SERVO_CONFIG` in those services as well.

Install the capture environment:

```bash
bash scripts/install_lerobot.sh
```

This installs LeRobot v0.6.1 and the G1 adapter into `.venv_lerobot` using uv.
For SONIC, prepare `.venv_sim`, `.venv_data_collection`, `.venv_teleop` and the
TensorRT/CUDA stack according to the controller setup. System driver and service
installation is covered by the upstream guides in the asset catalog.

Build the controller from the local `GR00T-WholeBodyControl/` overlay, which
includes G1-specific changes. Training uses a separate source revision in
[`sources.toml`](../../third_party/sources.toml).

### Local hand-service configuration

Install a compatible dual-hand BrainCo DDS service separately on the robot.
Use the service supplied for your hardware and qualified for the G1 DDS command
and measured-state protocol. See the [hardware asset catalog](../../docs/public_assets.md#models-data-and-hardware-runtime).
Before teleoperation or policy execution, verify that left and right feedback
match the two hands. The launcher checks for fresh feedback from both sides.

For launcher-managed startup, set `hand_server_path` in local
`config/local.json` to the installed executable. Provide the verified hardware
serials in local `config/hand-hardware.json` when the service requires them.

For a custom CH346C adapter installation, start from
[`60-zeta0-brainco.rules.example`](scripts/hand_server/60-zeta0-brainco.rules.example).
Verify the USB adapter serial and left/right interface mapping. Install the
binding helper `scripts/hand_server/bind_wch.py` at the path referenced by the
local rule and pass the matching serial. These rules require the target Linux
driver to be installed by the operator.


## Collection

```bash
# Preview startup
python scripts/start_sonic.py --dry-run
# Start control and record a task
bash start_sonic.sh
bash record_episode.sh demo_native --prompt "Place the object on the table."
# Resume the same dataset and task
bash record_episode.sh demo_native
# Check live camera and state inputs
bash record_episode.sh --check --seconds 15
```

Wait for READY, then start recording. Enter saves an episode, Esc saves and
exits, and Ctrl+C discards the unsaved segment.

The collector writes LeRobot v3 data with measured `observation.state` and
separate action channels. Cameras are sampled independently. Check timestamps,
gaps, hand masks and outcome labels before preparing training data through
[the native v3 adapter](../../docs/framework.md#inspect-captured-data).
Legacy v2.1 helpers are in `scripts/legacy_collection`; use the loader matching
your dataset format.

## Deployment

Place model files in `.local/models/` and matching demonstration first frames in
`.local/first_frames/`. Fill in `config/model_paths.example.json` for your model
and runtime. The `tidy_scene` profile also needs its offline samples in
`.local/datasets/tidy_scene_offline/`.

Task launchers such as `bash deploy_place_chair.sh` start hardware control.
Complete calibration and emergency-stop checks before launching; startup
initializes the body and arms.

Press **f** to load the demonstration first frame, then **m** to start the model
after reference feedback. **v** returns to zero-velocity planning; **O** or
**Ctrl+C** requests a stop.

## Runtime compatibility

Zeta0 v0.2 uses the unified G1 runtime profile under `runtime_profiles/unified/`,
which loads directly from the maintained `src/psi` implementation and supports
both `zeta_native` (v5/v6, 30Hz/50 or 50Hz/30) and `gr00t_n1d7` (50Hz/40 or
30Hz/40) exports. See [`runtime_profiles/README.md`](runtime_profiles/README.md)
and [`COMPARISON.json`](runtime_profiles/COMPARISON.json).

Historical v5 task snapshots are archived on branch `archive/legacy-v5-runtimes`
and indexed in [`LEGACY_ARCHIVE.json`](runtime_profiles/LEGACY_ARCHIVE.json).
Profile selection is handled by `scripts/policy_profile.py` (`ZETA0_POLICY_PROFILE`,
default `unified`). An existing `.local/zeta0/<profile>` override takes
precedence and must match that profile's checksum manifest.

## Source and checks

From the repository root:

```bash
make test-g1
make check-release
```

See the [integration review](../../docs/maintenance/G1_INTEGRATION_REVIEW.md)
for source history, license review and hardware acceptance status.
[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records file provenance.
