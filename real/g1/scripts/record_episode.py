#!/usr/bin/env python3
"""Compatibility command name for the official LeRobot recorder."""
if __name__ == '__main__':
    from record_lerobot import main
    main()
else:
    # Retain historical helpers for offline verification. Executing this
    # command never imports or runs the legacy recording loop.
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module('legacy_collection.record_episode')
