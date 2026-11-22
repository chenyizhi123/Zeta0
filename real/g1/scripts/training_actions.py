"""Explicit hand action coordinates for datasets and offline inference adapters.

No control I/O. Finger target mean is preserved as legacy telemetry, never used
as a trigger coordinate. Projection residual makes off-curve DDS targets visible.
"""
import numpy as np
from brainco_controls import validate_grasp_config, grasp_command_speeds, hand_execution_policy
from recording_runtime import config_hash
from brainco_gripper_data import MissingGripperData


def training_action_features():
    return {
        'observation.hand_q':dict(dtype='float32',shape=(12,),names=[f'{s}_{i}' for s in ('left','right') for i in range(6)]),
        'action.hand_q':dict(dtype='float32',shape=(12,),names=[f'{s}_{i}' for s in ('left','right') for i in range(6)]),
        'action.hand_speed':dict(dtype='float32',shape=(12,),names=[f'{s}_speed_{i}' for s in ('left','right') for i in range(6)]),
        'action.grasp_profile':dict(dtype='int64',shape=(2,),names=['left_profile','right_profile']),
        'action.grasp_closure':dict(dtype='float32',shape=(2,),names=['left_curve_coordinate','right_curve_coordinate']),
        'action.grasp_projection_error':dict(dtype='float32',shape=(2,),names=['left_q_rmse','right_q_rmse']),
    }


def use_training_action_features(existing=None):
    if existing is None:return True
    present=[k in existing['features'] for k in training_action_features()]
    if any(present) and not all(present):raise ValueError('Incomplete explicit hand action schema')
    return all(present)


def decode_grasp_closure(config, profile, stroke):
    profiles,_=validate_grasp_config(config)
    if profile not in (0,1) or not np.isfinite(stroke) or not 0<=stroke<=1:
        raise ValueError('Expected profile 0/1 and closure coordinate 0..1')
    endpoint=np.asarray(profiles[profile],dtype=float)
    curve=config.get('closure_curve') or [dict(stroke=0.,fractions=[0]*6),dict(stroke=1.,fractions=[1]*6)]
    for low,high in zip(curve,curve[1:]):
        if stroke<=high['stroke']:
            alpha=(stroke-low['stroke'])/(high['stroke']-low['stroke'])
            fractions=np.asarray(low['fractions'])+alpha*(np.asarray(high['fractions'])-low['fractions'])
            return (endpoint*fractions).tolist()
    return endpoint.tolist()


def project_grasp_command(config, profile, q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all() or not ((0<=q)&(q<=1)).all():
        raise MissingGripperData('Invalid actual hand joint targets')
    validate_grasp_config(config)
    curve=config.get('closure_curve') or [dict(stroke=0.),dict(stroke=1.)]
    best=None
    for low,high in zip(curve,curve[1:]):
        lo=np.asarray(decode_grasp_closure(config,profile,low['stroke']))
        hi=np.asarray(decode_grasp_closure(config,profile,high['stroke']))
        direction=hi-lo;norm=float(direction@direction)
        alpha=float(np.clip((q-lo)@direction/norm,0,1)) if norm>0 else 0.
        stroke=low['stroke']+alpha*(high['stroke']-low['stroke'])
        error=float(np.sqrt(np.mean((q-(lo+alpha*direction))**2)))
        candidate=(error,stroke)
        if best is None or candidate<best:best=candidate
    return dict(closure=best[1],rmse=best[0])


def training_action_values(sample, grasp_config):
    if grasp_config is None:raise MissingGripperData('Loaded grasp configuration unavailable')
    profiles=sample.get('profile')
    if profiles is None or len(profiles)!=2:raise MissingGripperData('Grasp profiles unavailable')
    arrays={}
    for field in ('state_joints','action_joints','action_speed_joints'):
        value=np.asarray(sample.get(field),dtype=float)
        if value.shape!=(2,6) or not np.isfinite(value).all() or not ((0<=value)&(value<=1)).all():
            raise MissingGripperData('Actual telemetry unavailable: '+field)
        arrays[field]=value
    projected=[project_grasp_command(grasp_config,profiles[side],q) for side,q in enumerate(arrays['action_joints'])]
    return {
        'observation.hand_q':arrays['state_joints'].reshape(12).astype(np.float32),
        'action.hand_q':arrays['action_joints'].reshape(12).astype(np.float32),
        'action.hand_speed':arrays['action_speed_joints'].reshape(12).astype(np.float32),
        'action.grasp_profile':np.asarray(profiles,dtype=np.int64),
        'action.grasp_closure':np.asarray([v['closure'] for v in projected],dtype=np.float32),
        'action.grasp_projection_error':np.asarray([v['rmse'] for v in projected],dtype=np.float32),
    }


def decode_model_action(action, action_spec, deployment_contract, *, previous_hand_q=None):
    """Pure deployment mapping. The caller must own the single robot publisher.

    This function validates the recorded model/hand contract and returns targets;
    it never enables control, publishes commands, or performs robot resets.
    Compact actions keep a fixed recorded grasp profile and reuse its speed
    policy. Position-only actions use recorded execution speeds; 88-D actions
    preserve predicted q/dq. The publisher must apply FeedbackSlew at the returned
    rate, starting from fresh actual feedback. Tokens require matching SONIC artifacts.
    """
    values=np.asarray(action,dtype=float)
    space=action_spec.get('space')
    dimensions={'sonic-hand-position':76,'sonic-gripper':66,'sonic-hand-joints':88,'joint-gripper':31}
    if space not in dimensions or values.shape!=(dimensions[space],) or not np.isfinite(values).all():
        raise ValueError('Model action has an invalid interface, shape, or nonfinite value')
    config=deployment_contract.get('grasp_config')
    validate_grasp_config(config)
    if config_hash(config)!=deployment_contract.get('grasp_config_sha256'):
        raise ValueError('Deployment grasp mapping differs from the training contract')
    ceiling=deployment_contract.get('hand_speed_ceiling')
    if isinstance(ceiling,bool) or not isinstance(ceiling,(int,float)):
        raise ValueError('Recorded hand speed ceiling missing or invalid')
    # Also validates the ceiling and configured coordination ratios.
    grasp_command_speeds(config,ceiling)
    rate=deployment_contract.get('hand_target_rate_per_s',hand_execution_policy(config)['target_rate_per_s'])
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not np.isfinite(rate) or not 0<rate<=10:
        raise ValueError('Recorded hand target slew rate missing or invalid')
    body_count=29 if space=='joint-gripper' else 64
    if space=='sonic-hand-joints':
        q=values[64:76].reshape(2,6)
        speed=values[76:88].reshape(2,6)
        if not ((0<=q)&(q<=1)).all() or not ((0<=speed)&(speed<=ceiling+1e-6)).all():
            raise ValueError('Predicted finger command exceeds normalized bounds or the recorded speed ceiling')
    else:
        if space=='sonic-hand-position':
            q=values[64:76].reshape(2,6)
            if not ((0<=q)&(q<=1)).all():
                raise ValueError('Predicted finger position exceeds normalized bounds')
        else:
            profile=action_spec.get('fixed_grasp_profile')
            if profile not in ('power','tripod') or not ((0<=values[-2:])&(values[-2:]<=1)).all():
                raise ValueError('Compact action requires the fixed profile and closure coordinates in 0..1')
            q=np.asarray([decode_grasp_closure(config,1 if profile=='power' else 0,v) for v in values[-2:]])
        previous=None if previous_hand_q is None else np.asarray(previous_hand_q,dtype=float)
        if previous is not None and (previous.shape!=(2,6) or not np.isfinite(previous).all()):
            raise ValueError('Previous hand target must be a normalized 2x6 array')
        speed=np.asarray([grasp_command_speeds(config,ceiling,q[side],
            previous[side] if previous is not None else None) for side in range(2)])
    return dict(body_key=action_spec['body_key'],body_target=values[:body_count].tolist(),
                hand_q=q.tolist(),hand_speed=speed.tolist(),
                hand_target_rate_per_s=float(rate),
                grasp_config_sha256=deployment_contract['grasp_config_sha256'],
                execution_adapter='joint target adapter' if space=='joint-gripper' else 'SONIC motion-token decoder')
