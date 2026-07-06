#!/usr/bin/env python

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
"""Configuration for keyboard teleoperators."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig

    
@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""

    # TODO(Steven): Consider setting in here the keys that we want to capture/listen


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """Configuration for keyboard end-effector teleoperator.

    Used for controlling robot end-effectors with keyboard inputs.

    Attributes:
        use_gripper: Whether to include gripper control in actions
    """

    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_rover")
@dataclass
class KeyboardRoverTeleopConfig(TeleoperatorConfig):
    """Configuration for keyboard rover teleoperator.

    Used for controlling mobile robots like EarthRover Mini Plus with WASD controls.

    Attributes:
        linear_speed: Default linear velocity magnitude (-1 to 1 range for SDK robots)
        angular_speed: Default angular velocity magnitude (-1 to 1 range for SDK robots)
        speed_increment: Amount to increase/decrease speed with +/- keys
        turn_assist_ratio: Forward motion multiplier when turning with A/D keys (0.0-1.0)
        angular_speed_ratio: Ratio of angular to linear speed for synchronized adjustments
        min_linear_speed: Minimum linear speed when decreasing (prevents zero speed)
        min_angular_speed: Minimum angular speed when decreasing (prevents zero speed)
    """

    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    turn_assist_ratio: float = 0.3
    angular_speed_ratio: float = 0.6
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.05


@TeleoperatorConfig.register_subclass("keyboard_piper_joint")
@dataclass
class KeyboardPiperJointTeleopConfig(TeleoperatorConfig):
    """Keyboard teleoperator that directly increments Piper joint angles.

    Key mapping (6 joints + gripper):
        joint_1 : Q / A  (+/-)
        joint_2 : W / S  (+/-)
        joint_3 : E / D  (+/-)
        joint_4 : R / F  (+/-)
        joint_5 : T / G  (+/-)
        joint_6 : Y / H  (+/-)
        gripper : Z=open  X=close
        Step    : +/- to scale joint_step and gripper_step
    """

    # Step sizes per key event (degrees)
    joint_step: float = 1.0
    gripper_step: float = 5.0
    step_scale: float = 1.5


@TeleoperatorConfig.register_subclass("keyboard_piper_tcp")
@dataclass
class KeyboardPiperTCPTeleopConfig(TeleoperatorConfig):
    """Keyboard teleoperator that controls Piper TCP pose (position + orientation).

    Key mapping:
        Translation : W/S=+/-x  A/D=+/-y  Q/E=+/-z
        Rotation    : I/K=+/-pitch  J/L=+/-yaw  U/O=+/-roll
        Gripper     : Z=open  X=close
        Step resize : +/- to scale pos_step and rot_step

    action_features matches PIPER_ACTION_KEYS (joint positions).
    IK conversion is a stub until _tcp_to_joints() is implemented.
    """

    # Initial TCP pose: [x(m), y(m), z(m), roll(deg), pitch(deg), yaw(deg)]
    init_tcp_pose: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    # Initial gripper opening in degrees
    init_gripper_pos: float = 0.0

    # Step sizes per key event
    pos_step: float = 0.005  # metres
    rot_step: float = 1.0  # degrees
    gripper_step: float = 5.0  # degrees
    step_scale: float = 1.5  # multiplier applied by +/- keys
