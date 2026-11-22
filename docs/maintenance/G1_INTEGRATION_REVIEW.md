# G1 integration review — 2026-11-16

This historical review covers the supplied G1 source archive, compatibility,
private configuration, licenses and offline regression checks. The scope is
source integration and does not establish physical performance.

## Source provenance

The G1 tools are integrated from a collection and deployment source archive.
[SOURCE_MANIFEST.json](../../real/g1/SOURCE_MANIFEST.json) records its digest,
source file hashes, integrated file hashes and excluded dependencies.

## Changes and compatibility

| Boundary | Decision |
| --- | --- |
| Shared training code | Preserve `src/psi`, existing trainers/configs/checkpoint semantics and both hosts |
| G1 scripts | Keep names and relative layout under `real/g1`; scripts run from that directory |
| Frozen runtime | Track all five profiles outside ignored `.local`; keep original 46-file checksum maps |
| Runtime comparison | 35 identical paths, 11 differing paths; retain variants rather than asserting equivalence |
| Local runtime override | Existing `.local/zeta0/<profile>` takes precedence; hash/origin checks remain active |
| Private machine config | Omit original network, Wi-Fi, serial IDs, calibration, udev and systemd site files; supply placeholders |
| Embedded serial defaults | Replace original device identities with `REPLACE_` markers; new hardware must set validated identities |
| Configuration generator | Create only local config and logs; refuse overwrite; do not copy calibration |
| Dry-run | Teleoperation plan can run without private config or hardware libraries |
| Missing body provenance helper | Reconstruct the thin CLI against the supplied `write_body_runtime_manifest` signature; test actual hashing |
| Optional hardware files | Keep local native source changes; restore SDK/driver/geometry via explicit verified local import |
| Collection environment | Preserve separate Python 3.12/official LeRobot pin, outside Python 3.10 training lock |
| Wheel | Continue to contain only the maintained `psi` library; full hardware sources use clone/source export |

Source checks cover the exact release selection, including every recorded G1
source file. A regression test prevents `data/` ignore rules from accidentally
excluding runtime Python packages. The original integration retained both
incoming and integrated source digests and the original runtime manifests.
See [runtime compatibility](../../real/g1/README.md#runtime-compatibility) for
the current release profiles.

## Configuration and local assets

The release provides example configurations. Supply device addresses, camera
serials, calibrations and service settings for your own setup. Keep those files
with local models, recordings and logs. The release checks cover local artifact
boundaries and known credential patterns.

## Third-party inventory

| Component | Evidence retained | Distribution decision |
| --- | --- | --- |
| Psi0-derived library | Root notice, full Apache-2.0 text, upstream README | Existing attribution preserved |
| Supplied GR00T-WholeBodyControl overlay | Original complete dual source/model `LICENSE`; per-file source hashes | Source overlay only; model weights remain external; do not equate this modified snapshot with the training source pin |
| Official LeRobot RTC helper | Apache-2.0 file header and full adjacent `LICENSE` | Retain single helper and notice |
| Official LeRobot capture | Exact commit in installer; installed upstream retains license | Download separately; not a second vendored training stack |
| Five inference runtimes | Original `BASE_RUNTIME_LICENSE`, unchanged compatibility/hash records | Preserve historical source and attribution; base notices alone do not settle every model/data right |
| Unitree SDKs and nested DDS libraries | Archive per-file hashes; original license files restored only locally | Excluded from Git/source export; inspect original SDK and nested license texts before redistribution |
| XRoboToolkit source/bindings | Archive hashes and local license file | Excluded; platform binaries remain external |
| Robot-side BrainCo DDS service, vendor CH343 drivers | Original vendor source has separate terms | Install separately on the robot. This repository includes the DDS client and a configurable USB binding example. |
| Robot geometry/meshes | Archive hashes; no independent asset-rights determination | Excluded; retain locally only as supplied |
| BrainCo grasp example | Supplied MIT notice in excluded original subtree | Local restore only; not silently relicensed |

This is an inventory of declarations and boundaries, not a legal opinion.
No new license is invented for a dependency. Root notices do not relicense
external source, assets, models, training data or hardware SDKs.

## Validation and remaining acceptance

`make test-g1` checks archived source identities, frozen inference runtimes,
profile isolation and local configuration handling without connecting to a robot.
CI runs the G1 checks on Python 3.10 and 3.12. Native video decoding tests use
FFmpeg. The [workflow guide](../workflows.md#validation-status) describes
the scope of the checks and the assets needed for hardware validation.
