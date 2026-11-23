.DEFAULT_GOAL := help
PYTHON ?= python
RUFF ?= ruff

.PHONY: help check check-docs check-release test-dev test-cpu test-mainline test-gr00t lint audit

help:
	@printf '%s\n' 'check          docs, tooling, mainline CPU contracts and correctness lint' 'check-docs     maintained documentation links (offline)' 'test-dev       development and source export tests' 'test-cpu       bounded predictive-world checks; no GPU' 'test-mainline  full predictive-world regressions; needs training/robot dependencies' 'test-gr00t     native CPU contracts; pinned source required' 'check-release  export boundaries and G1 source identities' 'lint           correctness checks; no auto-fix' 'audit          source syntax and credential-pattern inventory (JSON)'

check:
	$(MAKE) test-g1
	$(MAKE) check-docs
	$(MAKE) test-dev
	$(MAKE) test-cpu
	$(MAKE) lint

check-docs:
	$(PYTHON) scripts/check_docs.py

check-release:
	$(PYTHON) scripts/check_release.py
	$(PYTHON) scripts/check_g1_sources.py

test-dev:
	CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/test_check_docs.py tests/test_export_eval_summary.py tests/test_development_config.py tests/test_distribution.py tests/test_repository_audit.py tests/test_json_config_audit.py tests/test_source_pins.py tests/test_public_setup.py tests/test_subprocess_cleanup.py tests/test_source_export.py tests/test_source_package.py tests/test_release.py tests/test_framework.py tests/test_oss_upgrade.py

test-cpu:
	PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q \
		tests/predictive_world/test_config.py \
		tests/predictive_world/test_contracts.py \
		tests/predictive_world/test_history.py \
		tests/predictive_world/test_latent_transition.py \
		tests/predictive_world/test_checkpoint.py \
		tests/predictive_world/test_dual_response_objective.py \
		tests/predictive_world/test_engine.py

test-mainline:
	PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/predictive_world

test-gr00t:
	$(PYTHON) scripts/check_source_pins.py Isaac-GR00T
	PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) -c 'from psi.models.predictive_world.gr00t import native_types; native_types()'
	PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q tests/predictive_world/test_gr00t.py tests/predictive_world/test_gr00t_assets.py tests/predictive_world/test_gr00t_ddp.py tests/predictive_world/test_gr00t_preparation.py tests/predictive_world/test_backend_release.py

lint:
	$(RUFF) check src/psi src/zeta0 scripts tests real/g1/scripts real/g1/adapters real/g1/configure_local_paths.py

audit:
	$(PYTHON) scripts/audit_repository.py

.PHONY: test-g1
test-g1:
	$(PYTHON) scripts/check_g1_sources.py
	$(PYTHON) -m pytest -q tests/test_g1_integration.py

.PHONY: demo test-gpu

demo:
	PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $(PYTHON) scripts/examples/quickstart.py

test-gpu:
	PYTHONPATH=src OMP_NUM_THREADS=1 $(PYTHON) scripts/validate_gpu.py --output runs/gpu-acceptance
