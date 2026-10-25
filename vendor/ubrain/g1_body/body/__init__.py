"""body -- the MuJoCo body for ubrain, generic over the robot.

The brain never imports this package. It talks to it across a process boundary
through the tick bridge (body/bridge.py) using the protocol in body/protocol.py;
body/client.py is the dependency-free reference client, which ubrain carries as
`ubrain.bodies`, byte for byte. Which robot is the body is a `RobotSpec`
from `robots/` (`robots/g1` first); nothing under `body/` names a robot.

Modules:
  robot       RobotSpec and Primitive -- what a robot package must provide
  manifest    versions and digests that identify THIS body (into the cache key):
              simulator, body package, robot package, MuJoCo, menagerie, numpy
  sim         MujocoSim: load a spec's model, reset to its home keyframe,
              one control tick = 10 physics steps
  sensors     proprioception (IMU + joint state, laid out by the spec), optional
              contact, the camera; the privileged state the brain must never see
  primitives  the controller: an index and parameters in, joint targets out;
              the codebook itself is the spec's
  protocol    the message schema (pure stdlib)
  bridge      the tick server
  client      the reference client (pure stdlib)
"""

__version__ = "0.7.0"
