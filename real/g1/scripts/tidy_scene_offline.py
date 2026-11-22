"""Read recorded tidy_scene observations, without robot IO or model loading."""
import base64
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.local/datasets/tidy_scene_offline'


def requests(manifest):
    from psi.deploy.real_sonic_protocol import FORMAT, PROTOCOL, STATE_SCHEMA
    with np.load(DATA/'samples.npz',allow_pickle=False) as source:
        for seq,(index,state) in enumerate(zip(source['frame_indices'],source['states'])):
            yield dict(format=FORMAT,action_protocol_id=PROTOCOL,state_schema_id=STATE_SCHEMA,
                session_id='offline-tidy-scene',request_seq=seq,queue_revision=0,
                anchor_time=int(index)/30,task=manifest['task'],
                state=dict(values=[state.tolist()],valid=[[True]*41],times=[0.]),
                images={f'observation.images.{camera}':dict(
                    encoded=base64.b64encode((DATA/str(index)/(camera+'.png')).read_bytes()).decode(),time=0.)
                    for camera in ('ego_view','left_wrist','right_wrist')},committed_prefix=[])
