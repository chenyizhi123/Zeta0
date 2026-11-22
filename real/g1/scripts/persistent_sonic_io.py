"""Policy transport cleanup must never send the native whole-body STOP packet."""
from online_runtime import import_runtime

import_runtime()
from psi.deploy.real_sonic_io import SonicZmqIO


class PersistentSonicIO(SonicZmqIO):
    def close(self):
        if self.context.closed:
            return
        # Native lease expiry selects zero-velocity PLANNER. Only an explicit
        # operator exit handled by the process owner may stop the native body.
        self.active = False
        super().close()
