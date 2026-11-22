# Third-party notices

Zeta0 includes code inherited and adapted from Psi0 in its training library and
deployment runtimes. The inherited copyright and license notice are retained in `NOTICE`. The full
Apache-2.0 text is in `LICENSE` and `LICENSES/Apache-2.0.txt`. Original
project attribution remains in [the upstream README](docs/upstream/README.original.md).
The `psi` Python namespace does not change the authorship of inherited components.

The source distribution contains the maintained predictive-world implementation
and its shared data/image helpers. Unrelated inherited baseline trees were removed during the initial consolidation.
The later supplied G1 collection/deployment source is integrated under `real/g1`
with separate provenance and runtime identities.

Optional external integrations are pinned in `third_party/sources.toml`. Their licenses, assets and model terms apply independently:

- GR00T-WholeBodyControl distinguishes code and model-weight terms. The local
  `src/psi/deploy/sonic_wire.py` message builders are adapted from its GEAR-SONIC
  planner sender; the source's Apache-2.0 attribution and notice are retained.
- Isaac-GR00T is a separately downloaded, pinned NVIDIA source checkout. Its
  source notices are retained there; GR00T and Cosmos weights remain external
  model assets with their own terms.
- `real/LICENSE` preserves the inherited real-integration license notice.
- Dataset, pretrained Qwen and robot/controller weights are external dependencies
  with their own acquisition and redistribution terms.

No external source, model, dataset or robot asset is relicensed by the root notice.
See [source distribution](docs/maintenance/RELEASE_CHECKLIST.md).

## Supplied G1 collection/deployment source

- `real/g1/GR00T-WholeBodyControl/` is a GEAR-SONIC source overlay for G1.
  It retains the [upstream dual license](real/g1/GR00T-WholeBodyControl/LICENSE),
  [software notices](LICENSES/GEAR-SONIC-THIRD-PARTY-NOTICES.txt), and
  [third-party license inventory](LICENSES/GEAR-SONIC-THIRD-PARTY-ASSET-LICENSES.txt)
  from the [pinned upstream revision](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl/tree/d40376994ff094787591ab76a691c50b8e131786/legal).
  The overlay identifies files that Zeta0 modifies. Model weights are acquired
  separately under the applicable model license.
- The controller carries [cnpy's MIT license](LICENSES/cnpy-MIT.txt), with the
  copyright of Carl Rogers. The XML parser, PicoSHA2 and FindTensorRT files
  retain their embedded MIT notices.
- The GEAR-SONIC rotation helpers retain the original notices for
  [PyTorch3D](LICENSES/PyTorch3D-BSD-3-Clause.txt),
  [ACTOR](LICENSES/ACTOR-MIT.txt),
  [PHC](LICENSES/PHC-BSD-3-Clause-Clear.txt),
  [Kornia](LICENSES/Kornia-Apache-2.0.txt) and
  [Ceres Solver](LICENSES/Ceres-BSD-3-Clause.txt).
  `gear_sonic/trl/utils/rotation_conversion.py` attributes the PyTorch3D
  implementation via ACTOR. `gear_sonic/isaac_utils/rotations.py` identifies
  the PHC code. `gear_sonic/trl/utils/kornia_transform.py` identifies the
  Kornia and Ceres transformations.
- `real/g1/vendor/lerobot/latency_tracker.py` retains its official Apache-2.0
  header and complete license beside the file. The independent capture installer
  pins official LeRobot v0.6.1 and preserves its installed source notices.
- The G1 runtime profiles retain the Psi0 base notices. Each profile's
  compatibility manifest records its source hashes.
- Zeta0 includes a BrainCo DDS client and a local udev configuration helper.
  Obtain and install a compatible two-hand DDS service and SDK separately.
  Vendor drivers and robot mesh assets are also acquired separately under their
  applicable terms.

See [the G1 license and integration review](docs/maintenance/G1_INTEGRATION_REVIEW.md).
The root notice does not grant rights to third-party hardware, models or datasets.


Public G1 setup uses exact source revisions and a per-file SHA256 manifest,
without the author's private archive. The XRoboToolkit CMake overlay retains the
upstream MIT text under `real/g1/patches/xrobotoolkit/LICENSE`. SDK/geometry files
are acquired separately and remain excluded from the source release. See
[the public acquisition catalog](docs/public_assets.md). The library wheel carries
`NOTICE` along with the complete Apache license and third-party notices.
