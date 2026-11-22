# Zeta-0 development principles

This is the independent source repository for Zeta-0, with Psi0-derived
(`zeta_native`) and native GR00T N1.7 (`gr00t_n1d7`) hosts. Reuse the shared
representation, predictor, objectives and trainer. Keep host-specific action
computation and initialization in their owning adapters. The original Psi0
baseline is a separate model.

New joint recipes explicitly use v6/raw targets. Preserve the serialized names,
checkpoint identities and migration rules of historical artifacts. Keep deployable
inputs causal; future observations supervise training only. See [backend contracts](docs/backends.md).

## Change boundaries

- Trace the relevant implementation, producers, consumers and tests before editing.
  Deliver one complete, reviewable outcome; preserve unrelated work, including
  untracked files. Resolve material ambiguities before changing behavior.
- Define behavior once in its responsible module. Reusable model/data logic belongs
  in `src/psi`; launchers configure resources and invoke entry points. Prefer existing
  configuration and extension points for experiment variants. When these cannot
  express the requirement cleanly, justify extending or refactoring them; do not
  create duplicate training logic or parallel configuration systems.
- Add abstractions and dependencies for demonstrated needs, not hypothetical cases.
  Update dependency declarations and locks together; validate environment changes
  in isolation rather than mutating shared installations.
- Keep machine paths and resource allocation outside reusable computation. Library
  imports must not start jobs, contact services or initialize robot/GPU resources.

## Semantic contracts

- Define tensor shapes, dtypes, masks, units and temporal meaning at component
  boundaries. Trace ambiguous inputs to their producer; do not guess schemas or
  silently alter data/model semantics to make an error disappear.
- Keep deployable inputs causal. Use recorded future observations as supervision
  during training.
- For API, config, data or checkpoint changes, identify affected producers/consumers
  and preserve supported semantics or provide an explicit migration with tests.
  Retain compatibility only for explicitly supported experiments, checkpoints and
  callers. Upstream conventions are not permanent constraints; evolve them through
  explicit, tested migrations when needed.
- Changes to mathematical definitions, data distributions or training strategies are
  method changes; record them in experiment configs. Implementation-only refactors
  require behavioral equivalence checks, with justified numerical tolerances where
  needed. Preserve existing run artifacts and their interpretation; update affected
  contracts and documentation together.

## Evidence and delivery

- Historical experiment records remain external. Do not duplicate, rewrite or
  relabel their results in this source-only repository. Store new run artifacts in
  ignored output directories and publish only reviewed evidence.
- Specify an experiment's hypothesis, comparison, budget and stopping condition.
  Record the actual source, environment, resolved config and data/teacher identity
  with its results. Verify these in the process that runs the experiment.
- Test observable behavior at its owning boundary, including relevant failure cases.
  Bug fixes need regression coverage where feasible; integration changes need real
  boundary tests. Do not mock away the behavior under test or weaken checks to pass.
- Review the final changes for correctness, compatibility and unnecessary complexity,
  including untracked files. Run relevant checks and report their exact results,
  failures, skips and untested paths. Limit conclusions to the evaluated setup:
  smoke tests, training quality and closed-loop performance are different evidence.
- Keep actions within task authorization. Training submissions, robot operations,
  shared-data/environment changes and publication are not implicit development steps.
  Preserve private data, source attribution and third-party licenses.

## Read when relevant

- Cross-module changes: [architecture and responsibility map](docs/maintenance/ARCHITECTURE.md).
- Model, data, training, evaluation, checkpoint, runtime or deployment behavior:
  [research contracts](docs/maintenance/RESEARCH_CONTRACTS.md). Follow implementation
  references for the affected contracts, not every historical investigation.
- Environment setup: [environment guidance](CONTRIBUTING.md#environment).
- Validation: select [test tiers](CONTRIBUTING.md#test-tiers) for the change;
  the [bounded CPU check](CONTRIBUTING.md#cpu-checks) does not
  replace full-model or training validation.
- Publication: [release checklist](docs/maintenance/RELEASE_CHECKLIST.md) and
  [third-party notices](THIRD_PARTY_NOTICES.md).

## Maintaining this file

Keep only durable rules that change recurring development decisions. Before adding
a rule, name the concrete mistaken decision its absence would make more likely;
if none can be identified, omit it. Encode enforceable invariants in code and tests.
Keep parameter values, command recipes and incident histories with their owning
configs/docs. Replace obsolete or duplicate guidance; resolved bugs belong in
regression tests, not permanent incident-specific instructions.
