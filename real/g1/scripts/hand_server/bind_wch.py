#!/usr/bin/python3
"""Bind only the verified BrainCo USB adapter's two UART interfaces."""
import re
import subprocess
import sys
import syslog
from pathlib import Path

USB = Path('/sys/bus/usb/devices')
DRIVERS = Path('/sys/bus/usb/drivers')


def bind_interface(name, serial):
    if not serial or serial.startswith("REPLACE_"):
        raise ValueError("Set the verified USB adapter serial")
    if not re.fullmatch(r'[0-9]+-[0-9.]+:1\.[02]', name):
        raise ValueError('Unexpected USB interface: ' + name)
    interface = USB / name
    device = USB / name.split(':')[0]
    expected = {'idVendor': '1a86', 'idProduct': '55ec', 'serial': serial}
    if any((device / key).read_text().strip() != value for key, value in expected.items()):
        raise RuntimeError('USB adapter identity mismatch')
    owner = interface / 'driver'
    driver = owner.resolve().name if owner.exists() else None
    if driver == 'usb_ch343':
        return
    if driver not in (None, 'cdc_acm'):
        raise RuntimeError('Unexpected current driver: ' + driver)
    subprocess.run(['/sbin/modprobe', 'ch343'], check=True, timeout=5)
    if driver == 'cdc_acm':
        (DRIVERS / 'cdc_acm' / 'unbind').write_text(name)
    try:
        if not owner.exists():
            (DRIVERS / 'usb_ch343' / 'bind').write_text(name)
        if not owner.exists() or owner.resolve().name != 'usb_ch343':
            raise RuntimeError('WCH driver did not bind')
    except Exception:
        if not owner.exists():
            (DRIVERS / 'cdc_acm' / 'bind').write_text(name)
        raise


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: zeta0-bind-wch-hand USB_INTERFACE EXPECTED_USB_SERIAL')
    try:
        bind_interface(sys.argv[1], sys.argv[2])
    except Exception as error:
        syslog.syslog(syslog.LOG_ERR, 'zeta0-bind-wch-hand: ' + repr(error))
        raise
