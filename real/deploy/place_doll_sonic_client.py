#!/usr/bin/env python3
"""Connect real cameras and measured G1/BrainCo state to the policy service."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from psi.deploy.real_sonic_client import main

if __name__ == "__main__":
    main()
