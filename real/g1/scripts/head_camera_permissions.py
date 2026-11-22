"""Grant the head publisher access to its identity-verified D455 RGB node."""
import argparse
import grp
import json
import os
from pathlib import Path
import stat


def head_rgb_node(serial, *, sysfs_root='/sys/class/video4linux', dev_root='/dev', node_stat=os.stat):
    matches = []
    for node in Path(sysfs_root).glob('video*'):
        parent = (node/'device').resolve()
        ancestors = [parent, *parent.parents]
        usb = next((p for p in ancestors if (p/'idVendor').exists()), None)
        interface = next((p for p in ancestors if (p/'bInterfaceNumber').exists()), None)
        if usb is None or interface is None:
            continue
        try:
            identity = tuple((usb/key).read_text().strip().lower()
                             for key in ('idVendor', 'idProduct', 'serial'))
            if identity != ('8086', '0b5c', serial.lower()):
                continue
            if int((interface/'bInterfaceNumber').read_text().strip(), 16) != 3:
                continue
            if int((node/'index').read_text().strip()) != 0:
                continue
            device = Path(dev_root)/node.name
            info = node_stat(device, follow_symlinks=False)
            if not stat.S_ISCHR(info.st_mode):
                raise RuntimeError('Head RGB node is not a character device: ' + str(device))
            expected = (node/'dev').read_text().strip()
            if expected != f'{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}':
                raise RuntimeError('Head RGB node differs from sysfs: ' + str(device))
            matches.append(device)
        except (OSError, ValueError):
            continue  # A disappearing device is not a valid permission target.
    if len(matches) != 1:
        raise RuntimeError(f'Expected one verified D455 RGB node, found {len(matches)}')
    return matches[0]


def restore_permissions(serial, *, dry_run=False, **selection):
    device = head_rgb_node(serial, **selection)
    if not dry_run:
        os.chown(device, 0, grp.getgrnam('video').gr_gid, follow_symlinks=False)
        # Jetson's Python 3.8 lacks chmod(follow_symlinks=False). Selection
        # already rejects symlinks in the root-owned device directory.
        os.chmod(device, 0o660)
    return dict(device=str(device),usb_serial=serial,group='video',mode='0660',
                permissions_applied=not dry_run,capture_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-usb-serial', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore_permissions(args.expected_usb_serial, dry_run=args.dry_run)), flush=True)


if __name__ == '__main__':
    main()
