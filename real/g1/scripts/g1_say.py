#!/usr/bin/env python3
"""Use only the G1 audio RPC service; never initialize motion clients."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='?')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'config/local.json')
    parser.add_argument('--interface')
    parser.add_argument('--volume-only', action='store_true')
    args = parser.parse_args()
    if not args.volume_only and not args.text:
        parser.error('需要提供播报内容')
    config = json.loads(args.config.read_text())
    sdk = Path(config['sonic_root']) / 'external_dependencies/unitree_sdk2_python'
    sys.path.insert(0, str(sdk))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    ChannelFactoryInitialize(0, args.interface or config['robot_interface'])
    client = AudioClient()
    client.SetTimeout(3.0)
    client.Init()
    if args.volume_only:
        code, volume = client.GetVolume()
        print(json.dumps(dict(code=code, volume=volume), ensure_ascii=False), flush=True)
    else:
        code = client.TtsMaker(args.text, 0)
        print(json.dumps(dict(code=code, text=args.text, output='g1_speaker'),
                         ensure_ascii=False), flush=True)
    return 0 if code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
