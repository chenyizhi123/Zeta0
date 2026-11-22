# Run the core without robot assets

Train a small predictor on synthetic data, save a checkpoint, then reload it
and resume training. This example uses Zeta0's `TemporalLatentTransition` and
shared training engine. A CPU environment is sufficient; no datasets or model
downloads are needed.

Use Python 3.10 and an isolated environment:

```bash
uv venv --python 3.10 .venv-check
# Linux CPU; on macOS omit --index-url.
uv pip install --python .venv-check/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-check/bin/python --group cpu-test
make demo PYTHON=.venv-check/bin/python
```

Results are saved to `runs/quickstart-*`. Check `summary.json` for checkpoint
reload and resume results; the example saves at update 4 and resumes through
update 8. The directory also contains `predictor.safetensors` and trainer state.

To choose a new output directory:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-check/bin/python scripts/examples/quickstart.py \
  --device cpu --output runs/my-first-check
```

With a prepared CUDA environment, the same example accepts `--device cuda`.
For native GPU contract validation, first acquire the pinned GR00T source and
install the backend groups from [backend installation](backends.md#install), then:

```bash
python scripts/fetch_sources.py Isaac-GR00T --apply
make test-gpu PYTHON=/path/to/cuda-environment/bin/python
```

`test-gpu` checks the CUDA example, BF16 optimizer updates and native-head
checkpoint equivalence. Results go to `runs/gpu-acceptance/`. To repeat with a
new output directory, use `scripts/validate_gpu.py --output <new-directory>`.
See the [H200 test report](maintenance/CUDA_ACCEPTANCE_20261116.md) for results.

For real data, model preparation and G1 deployment, continue with
[the workflow contracts](workflows.md) and [the public asset catalog](public_assets.md).
