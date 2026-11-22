"""Offline compatibility for datasets produced by the retired collector."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module('legacy_collection.lerobot_backend')
