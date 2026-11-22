# Security and deployment boundaries

Zeta0 is research software, not a safety-certified robot controller. Passing
offline tests does not authorize physical execution. Keep robot interlocks,
workspace limits, emergency stops and operator supervision outside the learned
policy's control.

## Reporting

Do not put credentials, private datasets, exploitable endpoints or sensitive logs
in public issues. Use [GitHub private vulnerability reporting](https://github.com/chenyizhi123/Zeta0/security/advisories/new)
when the repository offers it. If that option is unavailable, open a
[contact request](https://github.com/chenyizhi123/Zeta0/issues/new) with the title
“Private security contact requested” and no vulnerability details; the maintainer
can then arrange a private channel. Do not include exploit details in that issue.

## Trust boundaries

- Treat checkpoints, pickle files and custom model code as executable inputs.
  Load only reviewed, trusted artifacts in isolated environments. A checksum proves
  identity, not trustworthiness; `safetensors` does not validate surrounding code.
- Serving and teleoperation examples are not internet-facing services. Require
  network isolation and reviewed authentication/access controls before exposing them.
- Never publish raw sensor data, credentials, machine-specific launch settings or
  model weights without the required provenance and authorization review.
- Treat saved environment summaries and experiment-tracker metadata as logs, not
  automatically sanitized release artifacts. The training launcher now fully
  redacts its tracked token/key values, but historical outputs may contain values
  or fragments. Review them privately before export; a code fix does not remove
  copies already written or uploaded. Revoke/rotate any credential confirmed exposed.
- Dependency installation, dataset conversion, training and robot operation have
  different effects. The local CPU checks do not exercise or certify the latter three.

See [release gates](docs/maintenance/RELEASE_CHECKLIST.md) for outstanding reviews.
