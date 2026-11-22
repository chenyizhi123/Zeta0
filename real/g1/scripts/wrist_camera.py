"""Select verified wrist RGB nodes by USB identity, never by video numbering."""


def realsense_sdk_devices():
    """Read firmware identities in a bounded helper; never start/reset streams."""
    import json
    import subprocess
    import sys
    program = r'''
import ctypes as C
import ctypes.util
import json
from pathlib import Path
paths = ['/opt/ros/noetic/lib/aarch64-linux-gnu/librealsense2.so',
         '/usr/lib/aarch64-linux-gnu/librealsense2.so',
         '/usr/local/lib/librealsense2.so']
path = next((p for p in paths if Path(p).is_file()), None)
path = path or ctypes.util.find_library('realsense2')
if path is None:
    print('[]')
    raise SystemExit(0)
lib = C.CDLL(path)
V, I = C.c_void_p, C.c_int
E = C.POINTER(V)
def bind(name, args, result):
    f = getattr(lib, name); f.argtypes = args; f.restype = result
    return f
api = bind('rs2_get_api_version', [E], I)
create = bind('rs2_create_context', [I,E], V)
query = bind('rs2_query_devices', [V,E], V)
count = bind('rs2_get_device_count', [V,E], I)
device = bind('rs2_create_device', [V,I,E], V)
supports = bind('rs2_supports_device_info', [V,I,E], I)
info = bind('rs2_get_device_info', [V,I,E], C.c_char_p)
message = bind('rs2_get_error_message', [V], C.c_char_p)
free_error = bind('rs2_free_error', [V], None)
delete_device = bind('rs2_delete_device', [V], None)
delete_list = bind('rs2_delete_device_list', [V], None)
delete_context = bind('rs2_delete_context', [V], None)
def call(f, *args):
    error = V(); result = f(*args, C.byref(error))
    if error.value:
        text = message(error).decode(errors='replace'); free_error(error)
        raise RuntimeError(text)
    return result
# rs2_camera_info values from the installed librealsense 2.50 headers.
fields = {'NAME':0, 'SERIAL_NUMBER':1, 'PHYSICAL_PORT':4,
          'USB_TYPE_DESCRIPTOR':9, 'FIRMWARE_UPDATE_ID':12}
context = listing = None
records = []
try:
    context = call(create, call(api)); listing = call(query, context)
    for index in range(call(count, listing)):
        d = call(device, listing, index)
        try:
            records.append({key:call(info,d,value).decode(errors='replace')
                            for key,value in fields.items() if call(supports,d,value)})
        finally:
            delete_device(d)
    print(json.dumps(records))
finally:
    if listing: delete_list(listing)
    if context: delete_context(context)
'''
    try:
        result = subprocess.run([sys.executable, '-c', program], capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError('RealSense SDK 身份查询未完成') from error
    if result.returncode:
        raise RuntimeError('RealSense SDK 身份查询失败: ' + result.stderr[-500:])
    records = json.loads(result.stdout)
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise RuntimeError('RealSense SDK 返回了无效身份记录')
    return records


def resolve_wrist_usb_identity(source, identity, usb_root):
    """USB2 D415 may omit iSerial; require its SDK firmware ID and USB address."""
    from pathlib import Path
    identity = dict(identity)
    if (not source.get('usb_serial') or identity.get('serial')
            or (identity.get('idVendor','').lower(), identity.get('idProduct','').lower())
               != ('8086','0ad3')):
        return identity
    usb = Path(usb_root)
    try:
        number = int((usb/'devnum').read_text().strip())
    except (OSError, ValueError):
        return identity
    port = usb.name + '-' + str(number)
    records = [record for record in realsense_sdk_devices() if record.get('PHYSICAL_PORT') == port]
    if len(records) > 1:
        raise RuntimeError('Ambiguous RealSense SDK identity for USB address ' + port)
    if not records:
        return identity
    record = records[0]
    if (record.get('FIRMWARE_UPDATE_ID') != source['usb_serial']
            or not record.get('SERIAL_NUMBER')):
        return identity
    try:
        if int((usb/'devnum').read_text().strip()) != number:
            return identity  # Replugged during the SDK query; do not reuse its result.
    except (OSError, ValueError):
        return identity
    identity.update(serial=record['FIRMWARE_UPDATE_ID'], usb_descriptor_serial=None,
                    serial_source='sdk_firmware_update_id', usb_device_number=number,
                    sdk_serial=record['SERIAL_NUMBER'], sdk_firmware_update_id=record['FIRMWARE_UPDATE_ID'],
                    sdk_physical_port=port, sdk_usb_type=record.get('USB_TYPE_DESCRIPTOR'))
    return identity


def wrist_camera_settings(source):
    vendor = source.get('usb_vendor_id', '1c45').lower()
    product = source.get('usb_product_id', '6200').lower()
    profiles = {('1c45', '6200'): ('MJPG', None),
                ('8086', '0ad3'): ('YUYV', '1.3')}
    if (vendor, product) not in profiles:
        raise ValueError('Unverified wrist camera USB identity: '+vendor+':'+product)
    pixel_format, interface = profiles[vendor, product]
    if source.get('pixel_format', pixel_format) != pixel_format:
        raise ValueError('Unsupported RGB format for this wrist camera')
    if source.get('usb_interface', interface) != interface:
        raise ValueError('Unexpected wrist RGB USB interface')
    if 'usb_serial' in source:
        serial = source['usb_serial']
        if not isinstance(serial, str) or not serial or serial != serial.strip():
            raise ValueError('Wrist USB serial must be an explicit non-empty string')
    return dict(vendor=vendor, product=product, pixel_format=pixel_format,
                interface=interface)


def wrist_identity_matches(source, identity):
    """A configured serial binds the camera across port moves; legacy uses path."""
    settings = wrist_camera_settings(source)
    if not isinstance(identity, dict):
        return False
    if identity.get('serial_source') == 'sdk_firmware_update_id':
        number = identity.get('usb_device_number')
        if (not isinstance(number, int) or isinstance(number, bool) or number <= 0
                or identity.get('usb_descriptor_serial') not in (None, '')
                or not identity.get('sdk_serial')
                or identity.get('sdk_firmware_update_id') != identity.get('serial')
                or identity.get('sdk_physical_port') != str(identity.get('usb_path'))+'-'+str(number)):
            return False
    binding_matches = (identity.get('serial') == source['usb_serial']
                       if 'usb_serial' in source
                       else identity.get('usb_path') == source['usb_path'])
    vendor, product = identity.get('idVendor'), identity.get('idProduct')
    return (binding_matches and
            isinstance(vendor, str) and vendor.lower() == settings['vendor'] and
            isinstance(product, str) and product.lower() == settings['product'])


def wrist_status_matches(status, name, source, now, publisher_pids):
    """Require fresh frames from the running publisher and the bound RGB device."""
    import math
    settings = wrist_camera_settings(source)
    if not isinstance(status, dict):
        return False
    usb = status.get('usb', {})
    stamp = status.get('last_frame_wall')
    frames = status.get('frames')
    pid = status.get('pid')
    session = status.get('source_session')
    return (status.get('source_key') == name
            and isinstance(pid, int) and not isinstance(pid, bool)
            and pid in publisher_pids
            and isinstance(session, str) and bool(session)
            and (status.get('width'), status.get('height')) == (640, 480)
            and wrist_identity_matches(source, usb)
            and (settings['interface'] is None or usb.get('usb_interface') == settings['interface'])
            and isinstance(frames, int) and not isinstance(frames, bool) and frames > 0
            and isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            and math.isfinite(stamp) and stamp > 0
            and isinstance(now, (int, float)) and math.isfinite(now)
            and 0 <= now - stamp <= 1.5)


def capture_capabilities(device):
    import fcntl
    import struct
    query = bytearray(104)
    with open(device, 'rb', buffering=0) as stream:
        fcntl.ioctl(stream, 0x80685600, query, True)
    flags = struct.unpack_from('I', query, 84)[0]
    return struct.unpack_from('I', query, 88)[0] if flags & 0x80000000 else flags


def wrist_capture_matches(source, identity, interface, index, capabilities):
    settings = wrist_camera_settings(source)
    return (wrist_identity_matches(source, identity) and
            (settings['interface'] is None or interface == settings['interface']) and
            index == 0 and bool(capabilities & 1))


def select_wrist_capture_device(source, sysfs_root='/sys/class/video4linux', dev_root='/dev'):
    from pathlib import Path
    settings = wrist_camera_settings(source)
    matches = []
    unavailable = []
    candidates = []
    for node in Path(sysfs_root).glob('video*'):
        parent = (node / 'device').resolve()
        usb = next((p for p in [parent, *parent.parents] if (p / 'idVendor').exists()), None)
        if usb is None:
            continue
        interface = next((p.name.split(':', 1)[1] for p in [parent, *parent.parents]
                          if p.parent == usb and ':' in p.name), None)
        device = str(Path(dev_root) / node.name)
        try:
            identity = {key: (usb / key).read_text().strip() for key in ('idVendor', 'idProduct')}
            identity['usb_path'] = usb.name
            if (usb / 'serial').exists():
                identity['serial'] = (usb / 'serial').read_text().strip()
            index = int((node / 'index').read_text().strip())
        except (OSError, ValueError):
            continue
        if ((identity['idVendor'].lower(), identity['idProduct'].lower())
                != (settings['vendor'], settings['product']) or index != 0
                or (settings['interface'] is not None and interface != settings['interface'])):
            continue
        candidates.append((device, identity, usb, interface, index))
    # A descriptor-verified camera does not depend on SDK enumeration of other
    # cameras. A USB2 sibling may time out in the SDK while this USB3 RGB node
    # is already identified. Keep every direct match so ambiguity still fails.
    direct = [row for row in candidates
              if wrist_capture_matches(source, row[1], row[3], row[4], 1)]
    if direct:
        candidates = direct
    for device, identity, usb, interface, index in candidates:
        identity = resolve_wrist_usb_identity(source, identity, usb)
        # Query only identity-verified RGB capture candidates. An inaccessible
        # depth/metadata node must not hide an otherwise usable RGB interface.
        if not wrist_capture_matches(source, identity, interface, index, 1):
            continue
        try:
            capabilities = capture_capabilities(device)
        except OSError as error:
            power = usb / 'power' / 'runtime_status'
            try:
                status = '; USB runtime_status=' + power.read_text().strip()
            except OSError:
                status = ''
            unavailable.append(device + ': ' + str(error) + status)
            continue
        if wrist_capture_matches(source, identity, interface, index, capabilities):
            matches.append(device)
    if unavailable:
        raise RuntimeError('腕部 RGB 已识别但无法访问（非 USB 路径缺失）: ' + '; '.join(unavailable))
    if len(matches) > 1:
        raise RuntimeError('Ambiguous wrist RGB devices: '+', '.join(matches))
    return matches[0] if matches else None


def wrist_publisher_args(script, name, source, robot_host, device):
    settings = wrist_camera_settings(source)
    args = ['python3', '-u', script, '--device', device, '--source-key', name,
            '--expected-usb-path', source['usb_path'],
            '--expected-usb-vendor', settings['vendor'],
            '--expected-usb-product', settings['product'],
            '--pixel-format', settings['pixel_format'],
            '--bind', 'tcp://'+robot_host+':'+str(source['port']),
            '--status', '/tmp/sonic-wrist-'+name+'-status.json']
    if settings['interface'] is not None:
        args += ['--expected-usb-interface', settings['interface']]
    if 'usb_serial' in source:
        args += ['--expected-usb-serial', source['usb_serial']]
    return args
