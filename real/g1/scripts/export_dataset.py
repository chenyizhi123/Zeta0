#!/usr/bin/env python3
"""Retired command name, forwarded to the supported official LeRobot entry."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    command = Path(__file__).resolve().parents[1] / 'record_episode.sh'
    os.execvp('bash', ['bash', str(command), *sys.argv[1:]])
