import subprocess
import threading
from types import SimpleNamespace
import pytest

from psi.data.predictive_world.performance import GpuTelemetry, nvidia_smi_identifier


@pytest.mark.parametrize('identifier', ['0', 'GPU-fixture', 'MIG-fixture', '00000000:3B:00.0'])
def test_existing_nvidia_device_identifiers_are_preserved(identifier):
    assert nvidia_smi_identifier(identifier) == identifier


def test_torch_bare_uuid_is_prefixed_for_nvidia_smi():
    identifier = '4e0f271b-3f54-e156-3723-1da30baff227'
    assert nvidia_smi_identifier(identifier) == 'GPU-' + identifier


def test_gpu_telemetry_samples_in_background_and_stops(monkeypatch):
    sampled = threading.Event()

    def query(command, **kwargs):
        assert command[command.index('-i') + 1] == 'GPU-fixture'
        sampled.set()
        return SimpleNamespace(stdout='GPU-fixture, 87, 1234, 81920\n')

    monkeypatch.setattr(subprocess, 'run', query)
    telemetry = GpuTelemetry('GPU-fixture', interval=0.01)
    try:
        assert sampled.wait(timeout=2)
        # _sample completion precedes publication under the lock; wait on the
        # thread's stop event between attempts without blocking any train thread.
        for _ in range(100):
            result = telemetry.snapshot()
            if result['gpu_telemetry_status'] == 'ok':
                break
            threading.Event().wait(0.01)
        assert result['gpu_utilization_percent'] == 87
        assert result['gpu_memory_used_mib'] == 1234
        assert result['gpu_sample_age_seconds'] >= 0
    finally:
        telemetry.close()
    assert not telemetry.thread.is_alive()


def test_missing_gpu_monitor_is_reported_without_stopping_training(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('no nvidia-smi')

    monkeypatch.setattr(subprocess, 'run', missing)
    telemetry = GpuTelemetry('GPU-fixture', interval=0.01)
    try:
        for _ in range(100):
            result = telemetry.snapshot()
            if result['gpu_telemetry_status'] == 'unavailable':
                break
            threading.Event().wait(0.01)
        assert result['gpu_telemetry_status'] == 'unavailable'
        assert 'gpu_utilization_percent' not in result
    finally:
        telemetry.close()
