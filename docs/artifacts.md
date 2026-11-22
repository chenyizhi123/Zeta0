# Required assets

Prepare models, datasets and runtime assets for the workflow you want to run.
The [asset catalog](public_assets.md) lists acquisition sources. Task-trained
Zeta checkpoints and private datasets are not currently published.

| Input | Required identity |
| --- | --- |
| Frozen VLM / N1.7 base | Local checkpoint, config/weight digests and processor identity |
| Demonstrations | Episode/task manifest, timestamps, state/action/body schemas and training split |
| Predictive index | Immutable row order, future-query alignment, registry hashes and temporal ABI |
| Feature cache | Exact backbone/processor/preprocessing identity and per-row index binding |
| Training checkpoint | Original source/config identity, update count and initialization/migration metadata |
| Controller and robot runtime | Pinned source, compatible controller models and execution protocol |

See [reproducibility](reproducibility.md), [backend preparation](backends.md) and
[G1 setup](../real/g1/README.md#install-and-configure).

## Historical records

To reproduce a result, record its resolved configuration, data and model
versions, checkpoint and evaluation protocol. Available checks are listed in the [support matrix](workflows.md).
