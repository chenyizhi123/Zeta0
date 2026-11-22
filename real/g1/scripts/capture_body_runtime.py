"""Hash controller artifacts before exec; does not start the controller."""
import argparse
from pathlib import Path
from recording_runtime import write_body_runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--deployment-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error('PID must be positive')
    write_body_runtime_manifest(args.output, args.deployment_root, args.pid)


if __name__ == '__main__':
    main()
