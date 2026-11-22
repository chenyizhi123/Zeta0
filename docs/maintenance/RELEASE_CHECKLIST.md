# Source distribution

Develop in this independent `Zeta0` repository. Publish only reviewed source;
local weights, datasets, cached features and raw experiment outputs stay ignored.
The inherited private experiment catalog remains external.

## Source checks

```bash
make check
make test-mainline
make test-gr00t
make check-release
uv lock --check
uv build --wheel
python scripts/check_distribution.py dist/*.whl
```

See [backend prerequisites](../backends.md#validation) and the
[validation status](../workflows.md#validation-status). `check-release`
checks the exact source export candidate set, including vendored text, for local
artifacts and known credential patterns.
The separate `make audit` inventories maintained Python/JSON/shell syntax and
machine-path references. Neither scan proves the absence of secrets or grants
redistribution rights.

## Independent source export

```bash
python scripts/export_source.py --output /tmp/zeta0-release-source
```

Use a new directory outside the checkout. The exporter includes current tracked
and unignored source, respects archive exclusions, refuses source symlinks and
oversized files, and writes a per-file SHA-256 manifest. It excludes `.git` and
never creates a remote or publishes code. Verify the exported bytes against the
manifest. Build the wheel from the export as a second packaging check.
Training/preparation provenance and the G1 exact-export regression require Git: use a normal clone, or initialize
and commit the exported source as a new local Git repository before those
operations. Wheel building and import checks do not require Git history.

Optional source checkouts are excluded and reacquired using the exact pins in
[`sources.toml`](../../third_party/sources.toml), without submodules. External
assets are acquired separately through the [asset catalog](../public_assets.md).

## Public reproducibility

Source readiness and experiment reproducibility are separate. Before claiming a
reproducible result, provide the reviewed asset acquisition catalog, original
resolved configs/checkpoint identities, training environment, evaluation protocol
and aggregate evidence. These inputs are currently [external](../artifacts.md).
The public distribution lives at [Zeta0](https://github.com/chenyizhi123/Zeta0). Use [the security reporting route](../../SECURITY.md) for sensitive issues.

Attribution remains in [LICENSE](../../LICENSE),
[third-party notices](../../THIRD_PARTY_NOTICES.md) and the
[upstream README](../upstream/README.original.md). Preserve applicable source,
model and dataset terms when assembling any broader release.


## G1 hardware integration

Read [the integration and license inventory](G1_INTEGRATION_REVIEW.md). Run
`make test-g1` before export. Every frozen runtime source must appear in the
export; local `.local`, calibration, SDK/driver and robot geometry must not.
The explicit local importer is not a redistribution grant. Resolve missing
third-party terms before bundling those excluded components in a future release.
Check the actual PR CI results, including Python 3.12 and FFmpeg video decoding;
source/CPU checks do not replace robot or experiment acceptance.


## Public-source acceptance

- Root `LICENSE` contains the full Apache-2.0 text; `NOTICE` preserves inherited
  copyright. Wheel checks require both files and third-party notices.
- `make demo` exercises the real predictor/trainer, model save/reload and optimizer
  resume without unpublished inputs. Its synthetic result is not a robot benchmark.
- `fetch_sources.py` acquires full pinned commits and preserves modified checkouts.
  `setup_g1_dependencies.py` restores 1,115 verified public dependency files without
  the private archive. Optional vendor binaries and task weights have explicit
  acquisition boundaries in [the public catalog](../public_assets.md).
- Hardware configuration remains local. The root license does not relicense
  external SDKs, models, robot geometry or datasets.
- `make test-gpu` is an additional explicit CUDA contract tier. A source release
  must report whether that tier ran; CPU checks must not be called GPU acceptance.
- Review the exact release commit, source archive and evidence before publishing a version.


## Portable archive

```bash
python scripts/package_source.py --output /tmp/zeta0-source.tar.gz
```

The destination must be new and outside the checkout. Packaging uses the same
reviewed export selection and includes its SHA256 manifest. Tar members use
UID/GID zero, empty owner/group names and timestamp zero; executable bits are
preserved. This avoids carrying workstation ownership into root-run Linux
containers, where mismatched ownership can make Git reject an extracted source
tree. Gzip metadata is normalized and identical source bytes/modes produce an
identical archive. Archive publication never replaces an existing output.
