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

import logging
import math
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardPiperJointTeleopConfig,
    KeyboardPiperTCPTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear.vel' and 'angular.vel' keys
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear.vel": linear_velocity,
            "angular.vel": angular_velocity,
        }


class KeyboardPiperTCPTeleop(KeyboardTeleop):
    """Keyboard teleoperator that drives Piper by incrementing a target TCP pose.

    Internal state: ``tcp_pose = [x, y, z, roll, pitch, yaw]`` (metres / degrees)
    and ``gripper_pos`` (degrees).

    ``get_action()`` outputs an end-effector pose. The robot action processor
    converts that pose to Piper joint targets with IK.

    Key mapping
    -----------
    Translation  : W/S = ±x   A/D = ±y   Q/E = ±z
    Rotation     : I/K = ±pitch   J/L = ±yaw   U/O = ±roll
    Gripper      : Z = open (+step)   X = close (-step)
    Step scale   : + / - to multiply / divide pos_step and rot_step
    """

    config_class = KeyboardPiperTCPTeleopConfig
    name = "keyboard_piper_tcp"

    # Keys that produce a TCP delta; anything else is ignored.
    _TRANSLATION_KEYS = {"w", "s", "a", "d", "q", "e"}
    _ROTATION_KEYS = {"i", "k", "j", "l", "u", "o"}
    _GRIPPER_KEYS = {"z", "x"}
    _STEP_KEYS = {"+", "=", "-"}

    def __init__(self, config: KeyboardPiperTCPTeleopConfig):
        super().__init__(config)
        self.config = config

        # Mutable TCP state. Config keeps the user-facing [x, y, z, roll, pitch, yaw] format.
        self._tcp_position: list[float] = list(config.init_tcp_pose[:3])
        self._tcp_rotvec: list[float] = self._rpy_degrees_to_rotvec(*config.init_tcp_pose[3:])
        self._gripper_pos: float = config.init_gripper_pos
        self._pos_step: float = config.pos_step
        self._rot_step: float = config.rot_step

    @staticmethod
    def _rpy_degrees_to_rotvec(roll: float, pitch: float, yaw: float) -> list[float]:
        roll_r, pitch_r, yaw_r = math.radians(roll), math.radians(pitch), math.radians(yaw)
        cr, sr = math.cos(roll_r), math.sin(roll_r)
        cp, sp = math.cos(pitch_r), math.sin(pitch_r)
        cy, sy = math.cos(yaw_r), math.sin(yaw_r)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        angle = math.acos(max(-1.0, min(1.0, (r00 + r11 + r22 - 1.0) / 2.0)))
        if abs(angle) < 1e-9:
            return [0.0, 0.0, 0.0]

        s = 2.0 * math.sin(angle)
        return [
            (r21 - r12) / s * angle,
            (r02 - r20) / s * angle,
            (r10 - r01) / s * angle,
        ]

    def set_tcp_target(
        self,
        x: float,
        y: float,
        z: float,
        wx: float,
        wy: float,
        wz: float,
        gripper_pos: float | None = None,
    ) -> None:
        self._tcp_position = [float(x), float(y), float(z)]
        self._tcp_rotvec = [float(wx), float(wy), float(wz)]
        if gripper_pos is not None:
            self._gripper_pos = max(0.0, min(100.0, float(gripper_pos)))

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.wx": float,
            "ee.wy": float,
            "ee.wz": float,
            "ee.gripper_pos": float,
        }

    @property
    def is_calibrated(self) -> bool:
        return True

    def _apply_key_deltas(self) -> None:
        """Drain the event queue and update internal TCP / gripper state."""
        self._drain_pressed_keys()

        for key, pressed in list(self.current_pressed.items()):
            if not pressed:
                continue
            k = key.lower() if isinstance(key, str) else key

            # Translation
            if k == "w":
                self._tcp_position[0] += self._pos_step
            elif k == "s":
                self._tcp_position[0] -= self._pos_step
            elif k == "a":
                self._tcp_position[1] += self._pos_step
            elif k == "d":
                self._tcp_position[1] -= self._pos_step
            elif k == "q":
                self._tcp_position[2] += self._pos_step
            elif k == "e":
                self._tcp_position[2] -= self._pos_step
            # Rotation
            elif k == "i":
                self._tcp_rotvec[1] += math.radians(self._rot_step)
            elif k == "k":
                self._tcp_rotvec[1] -= math.radians(self._rot_step)
            elif k == "j":
                self._tcp_rotvec[2] += math.radians(self._rot_step)
            elif k == "l":
                self._tcp_rotvec[2] -= math.radians(self._rot_step)
            elif k == "u":
                self._tcp_rotvec[0] += math.radians(self._rot_step)
            elif k == "o":
                self._tcp_rotvec[0] -= math.radians(self._rot_step)
            # Gripper
            elif k == "z":
                self._gripper_pos += self.config.gripper_step
            elif k == "x":
                self._gripper_pos -= self.config.gripper_step
            # Step scaling
            elif k in ("+", "="):
                self._pos_step *= self.config.step_scale
                self._rot_step *= self.config.step_scale
                logging.info(
                    "TCP step increased: pos=%.4fm  rot=%.2fdeg",
                    self._pos_step,
                    self._rot_step,
                )
            elif k == "-":
                self._pos_step /= self.config.step_scale
                self._rot_step /= self.config.step_scale
                logging.info(
                    "TCP step decreased: pos=%.4fm  rot=%.2fdeg",
                    self._pos_step,
                    self._rot_step,
                )

        # Clamp gripper to [0, 100] degrees (adjust if Piper range differs)
        self._gripper_pos = max(0.0, min(100.0, self._gripper_pos))

        # Key events are discrete increments; clear so each press fires once.
        self.current_pressed.clear()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        self._apply_key_deltas()
        x, y, z = self._tcp_position
        wx, wy, wz = self._tcp_rotvec
        return {
            "ee.x": x,
            "ee.y": y,
            "ee.z": z,
            "ee.wx": wx,
            "ee.wy": wy,
            "ee.wz": wz,
            "ee.gripper_pos": self._gripper_pos,
        }

    @property
    def tcp_pose(self) -> list[float]:
        """Current target TCP pose: [x, y, z, wx, wy, wz]."""
        return [*self._tcp_position, *self._tcp_rotvec]

    @property
    def gripper_pos(self) -> float:
        """Current target gripper opening in degrees."""
        return self._gripper_pos


class KeyboardPiperJointTeleop(KeyboardTeleop):
    """Keyboard teleoperator that directly increments Piper joint angles.

    Key mapping:
        joint_1 : Q / A  (+/-)
        joint_2 : W / S  (+/-)
        joint_3 : E / D  (+/-)
        joint_4 : R / F  (+/-)
        joint_5 : T / G  (+/-)
        joint_6 : Y / H  (+/-)
        gripper : Z=open  X=close
        Step    : +/- to scale joint_step and gripper_step
    """

    config_class = KeyboardPiperJointTeleopConfig
    name = "keyboard_piper_joint"

    # (pos_key, neg_key) for each joint in SDK order
    _JOINT_KEYS: list[tuple[str, str]] = [
        ("q", "a"),  # joint_1
        ("w", "s"),  # joint_2
        ("e", "d"),  # joint_3
        ("r", "f"),  # joint_4
        ("t", "g"),  # joint_5
        ("y", "h"),  # joint_6
    ]
    _GRIPPER_OPEN_KEY = "z"
    _GRIPPER_CLOSE_KEY = "x"
    _STEP_KEYS = ("+", "=", "-")

    def __init__(self, config: KeyboardPiperJointTeleopConfig):
        super().__init__(config)
        self.config = config
        self._joint_pos: list[float] = [0.0] * 6  # degrees, SDK convention
        self._gripper_pos: float = 0.0
        self._joint_step: float = config.joint_step
        self._gripper_step: float = config.gripper_step

    @property
    def action_features(self) -> dict[str, type]:
        from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES
        features = {f"{name}.pos": float for name in PIPER_JOINT_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def is_calibrated(self) -> bool:
        return True

    def _apply_key_deltas(self) -> None:
        self._drain_pressed_keys()
        for key, pressed in list(self.current_pressed.items()):
            if not pressed:
                continue
            k = key.lower() if isinstance(key, str) else key
            for i, (pos_k, neg_k) in enumerate(self._JOINT_KEYS):
                if k == pos_k:
                    self._joint_pos[i] += self._joint_step
                elif k == neg_k:
                    self._joint_pos[i] -= self._joint_step
            if k == self._GRIPPER_OPEN_KEY:
                self._gripper_pos += self._gripper_step
            elif k == self._GRIPPER_CLOSE_KEY:
                self._gripper_pos -= self._gripper_step
            elif k in ("+", "="):
                self._joint_step *= self.config.step_scale
                self._gripper_step *= self.config.step_scale
                logging.info("Joint step increased: joint=%.2fdeg  gripper=%.2fdeg", self._joint_step, self._gripper_step)
            elif k == "-":
                self._joint_step /= self.config.step_scale
                self._gripper_step /= self.config.step_scale
                logging.info("Joint step decreased: joint=%.2fdeg  gripper=%.2fdeg", self._joint_step, self._gripper_step)
        self._gripper_pos = max(0.0, min(100.0, self._gripper_pos))
        self.current_pressed.clear()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        self._apply_key_deltas()
        from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES
        action = {f"{name}.pos": self._joint_pos[i] for i, name in enumerate(PIPER_JOINT_NAMES)}
        action["gripper.pos"] = self._gripper_pos
        return action

    def set_joint_pos(self, joint_pos: list[float], gripper_pos: float = 0.0) -> None:
        """Seed the internal state from the robot's current joint positions."""
        self._joint_pos = list(joint_pos[:6])
        self._gripper_pos = float(gripper_pos)
