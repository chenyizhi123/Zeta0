"""Offline compatibility for the retired v2.1 video writer."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module('legacy_collection.episode_video_writer')
