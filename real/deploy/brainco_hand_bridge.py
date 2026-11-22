#!/usr/bin/env python3
"""Start a native BrainCo12 feedback/control bridge through the official SDK."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from psi.deploy.brainco_bridge import main

if __name__ == "__main__":
    main()
