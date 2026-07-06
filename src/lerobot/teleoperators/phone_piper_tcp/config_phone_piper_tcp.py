# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for phone WebXR teleoperator (Piper TCP control)."""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("phone_piper_tcp")
@dataclass
class PhonePiperTCPTeleopConfig(TeleoperatorConfig):
    """Configuration for phone WebXR teleoperator controlling Piper via TCP.

    The phone acts as a 6-DOF teleoperator: position + orientation from WebXR
    are mapped to the robot's end-effector target. The lerobot action processor
    pipeline converts the EE pose to joint angles via IK.

    Usage:
        lerobot-teleoperate \\
            --robot.type=piperx_follower \\
            --robot.port=can0 \\
            --teleop.type=phone_piper_tcp \\
            --teleop.port=5000
    """

    # Flask server port (shown in connect log message only, actual web server
    # is in phone_server.py; kept here for display purposes)
    port: int = 5000

    # ZMQ PUB address published by phone_server.py
    zmq_host: str = "localhost"
    zmq_port: int = 5556

    # Control frequency (Hz) – informational only for this client
    fps: int = 30
    control_freq: float = 30.0

    # Maximum acceptable message age in milliseconds (used by phone_server.py,
    # not this client directly; kept for documentation purposes)
    message_timeout_ms: float = 500.0
