#!/usr/bin/env python3
"""Collect verified G1 camera identities using SONIC's LeRobot exporter."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

import tyro
from brainco_gripper_data import adapt_schema, install_gripper_collector

root = Path(__file__).resolve().parents[2]
local = json.loads((root/'config/local.json').read_text())
path = Path(local['sonic_root'])/'gear_sonic/scripts/run_data_exporter.py'
spec = importlib.util.spec_from_file_location('chenyz_sonic_exporter', path)
sonic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sonic
spec.loader.exec_module(sonic)
if local['wrist_mapping'] != {'video6':'left_wrist', 'video8':'right_wrist'}:
    raise ValueError('Confirm wrist camera identities before exporting a dataset')
defaults = sonic.SonicDataExporterConfig(root_output_dir=str(root/'data'),
    camera_host='127.0.0.1', camera_port=local['camera_output_port'],
    data_collection_frequency=local['dataset_frequency'], task_prompt=local['task_prompt'],
    text_to_speech=False, record_wrist_cameras=True, robot_config_timeout=120)
config = tyro.cli(sonic.SonicDataExporterConfig, default=defaults)
if config.dataset_name is None:
    config.dataset_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
metadata = dict(configuration=local, hardware_synchronized=False,
                timestamp_kind='workstation_receive_time', wrist_mapping_verified=True,
                gripper=dict(hardware='BrainCo Revo2Touch', range=[0,1],
                             direction='0=open, 1=closed', order=['left','right'],
                             state_reduction='mean of six normalized joint feedback values',
                             action_source='actual six-channel DDS command, reduced by mean',
                             max_message_age_seconds=.2,
                             legacy_dex3_hand_fields_removed=True))
# Keep FK's existing robot model internally; export only the 29 body joints
# plus two separate scalar grippers, without neutral Dex3 placeholders.
model = sonic.get_g1_robot_model()
features, modalities = adapt_schema(sonic.get_features_sonic_vla(model),
                                   sonic.get_modality_config_sonic_vla(model), model)
sonic.get_features_sonic_vla = lambda robot_model: features
sonic.get_modality_config_sonic_vla = lambda robot_model: modalities
install_gripper_collector(sonic)
# LeRobot treats an existing dataset directory as a resume request. Let its
# factory create the directory before adding the session description.
original_create = sonic.Gr00tDataExporter.create
def create_with_metadata(*args, **kwargs):
    destination = Path(kwargs['save_root'])
    info = destination/'meta/info.json'
    if info.exists():
        existing = json.loads(info.read_text()).get('features', {})
        for key, expected in kwargs['features'].items():
            if key not in existing or existing[key]['dtype'] != expected['dtype'] or tuple(existing[key]['shape']) != tuple(expected['shape']):
                raise ValueError('Dataset schema differs; use a new dataset name for BrainCo grippers')
    kwargs['script_config'] = {**(kwargs.get('script_config') or {}), 'chenyz':metadata}
    exporter = original_create(*args, **kwargs)
    (exporter.root/'chenyz_session.json').write_text(json.dumps(metadata, indent=2))
    return exporter
sonic.Gr00tDataExporter.create = staticmethod(create_with_metadata)
sonic.main(config)
