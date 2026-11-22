## Purpose

What problem does this solve, and what behavior should the reviewer expect?
Link an existing issue if applicable; do not invent one.

## Scope and compatibility

- Affected components:
- Config/API/checkpoint/data compatibility and migration:
- Experiment or dependency changes (or none):
- Responsible module and affected representation/forecast/action/evaluation/deployment paths:

## Validation

List exact commands, environment and passed/failed/skipped results. Identify
unexecuted tests and why. For a bug, describe the reproducer/regression test.

For research changes, include the hypothesis/baseline, source and config identity,
data split/teacher provenance, tested hardware, resource use and result limitations.
State whether the workflow is exploratory, validated for a stated setup, or proposed
for maintenance. Do not infer closed-loop success from offline loss improvements.

## Risks and review focus

Describe remaining risks and any rollout/rollback considerations. Highlight
numerical, distributed, data-leakage or deployment changes when applicable.

## Checklist

- [ ] Change is scoped; unrelated user edits are preserved.
- [ ] Relevant tests and negative/error cases are covered.
- [ ] Documentation and versioned contracts match the final behavior.
- [ ] No credentials, private data or unreviewed artifacts are included.
- [ ] Applicable upstream attribution is preserved.

## Authorship and review status

- AI assistance: state whether used and its scope.
- Human review: pending unless an actual human reviewer has reviewed this change.

Completing this template does not constitute review approval or release permission.
