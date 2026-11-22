"""Pure conversion from native model actions to committed SONIC commands.

Shared by data adapters and the lightweight robot client.

The first 64 values are FSQ motion tokens; the last 12 are normalized BrainCo
targets. These targets precede the bridge's SDK rate limit and are not measured
hand positions. No hardware or runtime resources are initialized here.
"""

import numpy as np

from psi.real_observation_contract import validate_action_projection

MOTION_TOKEN_STEP = 1 / 16
MOTION_TOKEN_LIMIT = 10 / 16
ACTION_DIM = 76


def controller_command_values(actions, *, projection=None):
    """Return float32 commands of shape [76] or [N,76], without modifying input.

    Local clean_table tokens retain recorded [-0.75, 0.6875] bounds. A source-bound recorded projection
    preserves the training token range. BrainCo targets are clipped to [0,1]. The mapping is idempotent, so
    queued values, RTC snapshots and wire serialization share one value space.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != ACTION_DIM or not np.isfinite(values).all():
        raise ValueError("Controller actions must be finite with shape [76] or [N,76]")
    commands = values.copy()
    lower, upper = -0.75, 0.6875  # Supplied clean_table_trash 86-episode extrema.
    if projection is not None:
        profile = validate_action_projection(projection)
        lower, upper = profile["motion_token_min"], profile["motion_token_max"]
    commands[..., :64] = np.clip(
        np.round(np.clip(values[..., :64], lower, upper)
                 / MOTION_TOKEN_STEP) * MOTION_TOKEN_STEP,
        lower, upper,
    )
    commands[..., 64:] = np.clip(values[..., 64:], 0., 1.)
    return commands
