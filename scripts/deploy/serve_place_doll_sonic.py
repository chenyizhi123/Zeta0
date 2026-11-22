#!/usr/bin/env python3
"""Start the real-SONIC research inference service from a code-only checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from psi.deploy.real_sonic_server import main

if __name__ == "__main__":
    main()
