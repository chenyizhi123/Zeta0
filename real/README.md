# Real G1/BrainCo integration

The maintained policy uses state41/motion76, current collector snapshots,
body auxiliary supervision and FULL+ERT dual response v5. Training and cache
preparation follow [the final real-data recipe](../docs/real_data_final_method.md).

Robot-side dependencies are in `requirements-place-doll-robot.txt`; the client
uses HTTP plus existing SONIC camera/body/hand transport. Client and controller
configuration must match the checkpoint's advertised observation/action contracts.
The robot/controller source and hardware assets are external dependencies.

| Entry | Purpose |
| --- | --- |
| `deploy/place_doll_sonic_client.py` | Robot-side policy client |
| `deploy/brainco_hand_bridge.py` | Native BrainCo hand bridge |
| `../scripts/deploy/serve_place_doll_sonic.py` | Model-side research service |
| `../scripts/deploy/start_place_doll_sonic_controller.sh` | Controller launcher |
| `patches/` | Versioned external-controller protocol patch and provenance |

For live control, use the execution option with a compatible controller and
reference acknowledgements. Teleoperation, data collection and historical task
deployment use the separate [G1 stack](g1/README.md).
