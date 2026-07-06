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

"""Phone WebXR teleoperator for Piper robot arm – ZMQ client side.

This teleop is the *consumer* half of a two-process architecture:

  Process 1 (phone_server.py):
    Phone WebXR → Flask/SocketIO → TeleopController → placo IK → ZMQ PUB
    (also drives the real robot directly via CAN)

  Process 2 (lerobot-human-inloop-record):
    phone_piper_tcp.get_action() → ZMQ SUB → joint-space action
    → recording_loop (skips diff-IK because action is already joint-space)

Because the IK is done in phone_server.py using the same placo solver as
``teleop_service/real_robot.py``, there is no FK→IK round-trip error and no
diff-IK drift.  The recording loop only reads the pre-computed joint angles
and writes them to the dataset.

Usage
-----
Terminal 1 (phone server + robot driver):
    python teleop_service/phone_server.py --can-port can0 --no-calibrate

Terminal 2 (policy inference + dataset recording):
    lerobot-human-inloop-record \\
        --robot.type=piper_follower ... \\
        --teleop.type=phone_piper_tcp \\
        --policy.path=...
"""

import json
import logging
import threading
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_phone_piper_tcp import PhonePiperTCPTeleopConfig

logger = logging.getLogger(__name__)

# Keys published by phone_server.py in the ZMQ payload
_SDK_JOINT_KEYS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class PhonePiperTCPTeleop(Teleoperator):
    """ZMQ subscriber that consumes pre-computed joint angles from phone_server.py.

    ``get_action()`` returns a joint-space action dict::

        {joint_1.pos, joint_2.pos, ..., joint_6.pos, gripper.pos}

    Because the action is already in joint space the recording loop skips the
    ``robot_action_processor`` (diff-IK) automatically via the
    ``selected_from_policy`` guard in ``recording_loop.py``.
    """

    config_class = PhonePiperTCPTeleopConfig
    name = "phone_piper_tcp"

    def __init__(self, config: PhonePiperTCPTeleopConfig):
        super().__init__(config)
        self.config = config

        self._connected = False
        self._latest_action: dict[str, float] | None = None
        self._is_touching: bool = False   # True when phone primary_device_id is set
        self._lock = threading.Lock()
        self._sub_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def action_features(self) -> dict[str, type]:
        """Joint-space action – same keys as piper_follower.action_features."""
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the ZMQ PUB socket published by phone_server.py."""
        import zmq
        zmq_addr = f"tcp://{self.config.zmq_host}:{self.config.zmq_port}"
        logger.info("Connecting to phone_server ZMQ PUB at %s ...", zmq_addr)

        self._zmq_ctx = zmq.Context()
        self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sub.connect(zmq_addr)
        self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._zmq_sub.setsockopt(zmq.RCVTIMEO, 100)   # 100 ms receive timeout

        self._stop_event.clear()
        self._sub_thread = threading.Thread(
            target=self._recv_loop, name="phone-zmq-recv", daemon=True
        )
        self._sub_thread.start()

        self._connected = True
        logger.info(
            "phone_piper_tcp connected. Make sure phone_server.py is running "
            "and the phone browser is open at http://<host>:%d",
            self.config.port,
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._sub_thread is not None:
            self._sub_thread.join(timeout=1.0)
        try:
            self._zmq_sub.close()
            self._zmq_ctx.term()
        except Exception:
            pass
        self._connected = False
        logger.info("phone_piper_tcp disconnected.")

    # ------------------------------------------------------------------
    # ZMQ receive thread
    # ------------------------------------------------------------------
    def _recv_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self._zmq_sub.recv_string()
            except Exception:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("_type", "action")

            if msg_type == "action":
                action = {
                    f"{k}.pos": float(payload[k])
                    for k in _SDK_JOINT_KEYS
                    if k in payload
                }
                if "gripper" in payload:
                    action["gripper.pos"] = float(payload["gripper"])
                if len(action) == 7:
                    with self._lock:
                        self._latest_action = action
                        self._is_touching = True

            elif msg_type == "obs":
                # phone_server publishes obs every frame even when not touching.
                # We use this to detect when the phone stops touching.
                with self._lock:
                    self._is_touching = False

    # ------------------------------------------------------------------
    # Action / Observation
    # ------------------------------------------------------------------
    def set_observation(self, obs: dict[str, Any]) -> None:
        # Not needed: IK is computed in phone_server.py
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction | None:
        """Return the latest joint-space action from phone_server.

        Returns None when the phone is not actively touching, so the recording
        loop falls back to policy action (phone_server is not driving the robot
        either at that point).
        """
        with self._lock:
            if not self._is_touching or self._latest_action is None:
                return None
            return dict(self._latest_action)

    @property
    def is_phone_touching(self) -> bool:
        """True when the phone is actively touching the screen (driving the robot)."""
        with self._lock:
            return self._is_touching

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # Teleop events
    # ------------------------------------------------------------------
    def get_teleop_events(self) -> dict[str, Any]:
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }
