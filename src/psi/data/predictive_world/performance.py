"""Non-blocking, best-effort GPU telemetry for training performance logs."""
from __future__ import annotations

import subprocess
import threading
import time
from uuid import UUID


def nvidia_smi_identifier(identifier: str) -> str:
    # torch CUDA properties expose a bare UUID; NVML/nvidia-smi expect GPU-UUID.
    # Preserve numeric device IDs, PCI bus IDs and already-prefixed GPU/MIG IDs.
    try:
        return 'GPU-' + str(UUID(identifier))
    except ValueError:
        return identifier


class GpuTelemetry:
    def __init__(self, identifier: str, interval: float = 10):
        self.identifier = nvidia_smi_identifier(identifier)
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = {'gpu_telemetry_status': 'pending'}
        self.sampled_at = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _sample(self):
        result = subprocess.run([
            'nvidia-smi', '-i', self.identifier,
            '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=3, check=True)
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            raise ValueError('GPU identifier must select exactly one device')
        uuid, utilization, used, total = [part.strip() for part in rows[0].split(',')]
        return dict(gpu_telemetry_status='ok', gpu_uuid=uuid,
                    gpu_utilization_percent=float(utilization),
                    gpu_memory_used_mib=float(used), gpu_memory_total_mib=float(total))

    def _run(self):
        while not self.stop.is_set():
            try:
                sample = self._sample()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                sample = dict(gpu_telemetry_status='unavailable', gpu_telemetry_error=str(error))
            with self.lock:
                self.latest = sample
                self.sampled_at = time.monotonic()
            self.stop.wait(self.interval)

    def snapshot(self):
        with self.lock:
            result = dict(self.latest)
            if self.sampled_at is not None:
                result['gpu_sample_age_seconds'] = time.monotonic() - self.sampled_at
            return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)
