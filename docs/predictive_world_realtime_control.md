# Predictive-world asynchronous RTC control

This is an opt-in, client-side scheduling layer for a checkpoint trained with a
committed action prefix. It adopts the useful parts of LeRobot RTC—one control
loop, one background inference worker, raw versus postprocessed action queues,
measured latency, and late-reply rejection—without coupling `psi` to a robot
SDK. It is implemented in `psi.deploy.predictive_world_realtime`.

It is not a production robot approval, a safety controller, or an authorization
to start a hardware launcher. A robot integration must supply its own measured
observation builder, physical action sender, postprocessing/safety filter, and
an explicitly safe fallback action.

## Runtime contract

`RealtimeChunkController` owns two aligned queues:

* `raw`: unmodified semantic40 model actions. These form the next RTC prefix.
* `executable`: the corresponding postprocessed actions sent to the robot.

For a control observation at anchor `t`, the controller begins a request before
the action for that same tick is popped. Its prefix therefore starts at `t` and
contains the actions that can really execute while the request is in flight.
The response carries the full trained action horizon in `model_action_chunk`.
The controller discards exactly the measured number of elapsed control slots and
uses the remaining suffix only when all of the following hold:

1. session ID, session epoch, request ID, and queue revision still match;
2. the actual delay is no greater than the prefix reserved for that request;
3. that delay is within the checkpoint's trained RTC limit; and
4. both raw and postprocessed chunks have the expected finite shape.

An underestimated delay is rejected rather than silently switching to actions
conditioned on the wrong action history. Latency measurements remain visible in
`controller.metrics()` (`p50`, `p95`, maximum, rejected replies, deadline
misses, fallbacks, and queue depth). Planning uses the recent maximum latency,
not p95, as a conservative prefix reservation; p95 is reporting only.

## Startup and control-loop shape

For the current trained P30 / RTC6 checkpoint, use a 30-step semantic40 model
horizon, 50 Hz, and a maximum prefix of 6. `queue_low_watermark` must be at
least 6. Prime the queue before permitting motion, so the first motion does not
depend on an unmodelled fallback action.

```python
import numpy as np

from psi.deploy.predictive_world_realtime import (
    JsonHttpPolicyClient,
    RealtimeChunkController,
    RealtimeControlConfig,
)

client = JsonHttpPolicyClient("http://127.0.0.1:18096", timeout_s=1.0)
config = RealtimeControlConfig(
    action_hz=50.0,
    action_horizon=30,
    action_dim=40,
    max_rtc_delay_steps=6,
    queue_low_watermark=6,
    initial_delay_steps=6,
)
controller = RealtimeChunkController(
    config,
    client,
    remote_reset=client.reset,
    fallback_action=SAFE_SEMANTIC40_ACTION,  # supplied by the robot safety layer
    postprocess=robot_specific_safety_filter,
)

controller.start_session(seed=episode_seed)
controller.prime(build_observation(), anchor_time=measured_monotonic_seconds())
assert controller.wait_until_idle(timeout_s=2.0)  # before enabling robot motion

while control_is_authorized():
    observation, anchor = build_observation(), measured_monotonic_seconds()
    semantic40_action = controller.tick(observation, anchor_time=anchor)
    send_one_control_action(semantic40_action)
```

`build_observation()` must not add `committed_prefix` or `realtime`; the
controller creates those fields itself. `anchor_time` must use the same measured
clock domain as the state/image timestamp in the payload. The loop must call
`tick` exactly once per physical control period. On transport failure, queue
starvation, bad output, or a late reply, it returns only the caller-provided
fallback—never a fabricated zero action.
