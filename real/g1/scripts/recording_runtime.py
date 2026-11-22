"""Immutable recording provenance. No sockets, serial ports or control commands."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),
                                    allow_nan=False,ensure_ascii=False).encode()).hexdigest()


def current_boot_id():
    path=Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.exists() else 'unknown'


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as output:
            json.dump(value,output,ensure_ascii=False,indent=2,allow_nan=False)
            output.write('\n');output.flush();os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_runtime_manifest(path, kind, grasp_config, *, session=None, **extra):
    snapshot=copy.deepcopy(grasp_config)
    manifest=dict(format_version=1,kind=kind,pid=os.getpid(),boot_id=current_boot_id(),
                  session=session or uuid.uuid4().hex,started_wall=time.time(),
                  started_monotonic=time.monotonic(),grasp_config=snapshot,
                  grasp_config_sha256=config_hash(snapshot),**extra)
    atomic_json(path,manifest)
    return manifest


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_body_runtime_manifest(path, deployment_root, pid):
    deployment_root=Path(deployment_root).resolve()
    artifacts={}
    for name in ('target/release/g1_deploy_onnx_ref','policy/release/model_decoder.onnx',
                 'policy/release/model_encoder.onnx','policy/release/observation_config.yaml',
                 'planner/target_vel/V2/planner_sonic.onnx'):
        source=deployment_root/name
        artifacts[name]=dict(path=str(source),sha256=file_hash(source),bytes=source.stat().st_size)
    manifest=dict(format_version=1,kind='sonic_body',pid=int(pid),boot_id=current_boot_id(),
                  session=uuid.uuid4().hex,started_wall=time.time(),
                  artifacts=artifacts,identity_scope='files hashed immediately before shell exec; process PID retained by exec')
    atomic_json(path,manifest)
    return manifest


def decode_runtime_identity(manager):
    """Compact hc field: SHA256 bytes plus 16-byte process generation token."""
    values=manager.get('hc',[])
    if hasattr(values,'tolist'):values=values.tolist()
    if len(values)!=48 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or
                             not 0<=v<=255 or int(v)!=v for v in values):
        return None
    raw=bytes(int(v) for v in values)
    return dict(grasp_config_sha256=raw[:32].hex(),session=raw[32:].hex())


def runtime_process_alive(manifest, expected_name):
    if manifest.get('boot_id')!=current_boot_id():return False
    try:
        parts=(Path('/proc')/str(int(manifest['pid']))/'cmdline').read_bytes().split(b'\0')
        return any(Path(v.decode(errors='replace')).name==expected_name for v in parts if v)
    except (KeyError,OSError,TypeError,ValueError):return False


def capture_recording_runtime(local, manager, root=None, *, process_check=runtime_process_alive):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    blockers=[];manifests={}
    identity=decode_runtime_identity(manager)
    if identity is None:blockers.append('manager_runtime_identity_missing')
    for kind,filename,process in (
        ('manager','pico-hand-runtime.json','pico_manager.py'),
        ('bridge','brainco-bridge-runtime.json','brainco_bridge.py')):
        try:
            manifest=json.loads((root/'logs'/filename).read_text())
            if config_hash(manifest.get('grasp_config'))!=manifest.get('grasp_config_sha256'):
                raise ValueError('manifest config hash mismatch')
            manifests[kind]=manifest
            if not process_check(manifest,process):blockers.append(kind+'_runtime_process_not_verified')
        except (OSError,ValueError,TypeError):
            blockers.append(kind+'_runtime_manifest_missing_or_invalid')
    manager_manifest=manifests.get('manager',{})
    bridge_manifest=manifests.get('bridge',{})
    if identity and (identity['grasp_config_sha256']!=manager_manifest.get('grasp_config_sha256') or
                     identity['session']!=manager_manifest.get('session')):
        blockers.append('manager_runtime_generation_mismatch')
    if manager_manifest.get('grasp_config') is None:
        blockers.append('selectable_grasp_config_missing')
    if manager_manifest.get('grasp_config_sha256')!=bridge_manifest.get('grasp_config_sha256'):
        blockers.append('manager_bridge_grasp_config_mismatch')
    if bridge_manifest.get('publish') is not True:blockers.append('bridge_not_publishing')
    try:
        body_manifest=json.loads((root/'logs/sonic-body-runtime.json').read_text())
        if not process_check(body_manifest,'g1_deploy_onnx_ref'):
            blockers.append('body_runtime_process_not_verified')
    except (OSError,ValueError,TypeError):
        body_manifest={};blockers.append('body_runtime_manifest_missing')
    snapshot=copy.deepcopy(manager_manifest.get('grasp_config'))
    return dict(format_version=1,captured_wall=time.time(),captured_monotonic=time.monotonic(),
                grasp_config=snapshot,grasp_config_sha256=config_hash(snapshot) if snapshot else None,
                manager_runtime_manifest=copy.deepcopy(manager_manifest),
                bridge_runtime_manifest=copy.deepcopy(bridge_manifest),
                body_runtime_manifest=copy.deepcopy(body_manifest),
                recording_config=copy.deepcopy(local),recording_config_sha256=config_hash(local),
                runtime_verified=not blockers,blockers=blockers)


def runtime_sample_matches(sample, capture):
    expected=capture.get('manager_runtime_manifest',{})
    return (capture.get('runtime_verified') is True and
            sample.get('config_sha256')==capture.get('grasp_config_sha256') and
            sample.get('runtime_session')==expected.get('session'))
