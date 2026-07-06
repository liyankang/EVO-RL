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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import sys
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    IdentityProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,

)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_piper_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,

    phone_piper_tcp,
    piper_leader,

)
from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


LOOP_STATUS_INTERVAL_S = 0.5


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def _processor_pipeline_needs_observation(
    processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> bool:
    return not all(isinstance(step, IdentityProcessorStep) for step in processor.steps)


def _teleop_needs_robot_observation(
    display_data: bool,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> bool:
    return (
        display_data
        or _processor_pipeline_needs_observation(teleop_action_processor)
        or _processor_pipeline_needs_observation(robot_action_processor)
    )


def _configure_robot_for_lightweight_teleop(
    robot: Robot,
    should_fetch_obs: bool,
) -> None:
    if should_fetch_obs:
        return
    robot.set_teleop_send_only_mode(True)


def _teleop_robot_pair_needs_initial_observation(teleop: Teleoperator, robot: Robot) -> bool:
    return teleop.name == "keyboard_piper_tcp" and robot.name in {"piper_follower", "piperx_follower"}


def _make_keyboard_piper_tcp_robot_action_processor(
    robot: Robot,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    from lerobot.robots.piper_follower.robot_kinematic_processor import (
        PIPER_SDK_MOTOR_NAMES,
        PiperInverseKinematicsEEToJoints,
        make_piper_kinematics,
        piper_sdk_joints_to_urdf_joints,
    )
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
    )

    kinematics = make_piper_kinematics()

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.6, -0.6, 0.0], "max": [0.6, 0.6, 0.8]},
                max_ee_step_m=0.05,
            ),
            PiperInverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=PIPER_SDK_MOTOR_NAMES + ["gripper"],
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _configure_processors_for_teleop_robot_pair(
    teleop: Teleoperator,
    robot: Robot,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    if teleop.name == "keyboard_piper_tcp" and robot.name in {"piper_follower", "piperx_follower"}:
        logging.info("Using Piper keyboard TCP IK action processor.")
        return _make_keyboard_piper_tcp_robot_action_processor(robot)

    # phone_piper_tcp outputs joint-space actions (IK is done in phone_server.py),
    # so it does not need the diff-IK processor.
    if teleop.name == "phone_piper_tcp":
        logging.info("phone_piper_tcp uses pre-computed joint angles; skipping IK processor.")
        return robot_action_processor

    return robot_action_processor


def _sync_keyboard_piper_tcp_target_from_robot(
    teleop: Teleoperator,
    robot: Robot,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> None:
    if robot.name not in {"piper_follower", "piperx_follower"}:
        return

    observation = robot.get_observation()

    if teleop.name == "keyboard_piper_joint" and hasattr(teleop, "set_joint_pos"):
        from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES
        joint_pos = [float(observation.get(f"{name}.pos", 0.0)) for name in PIPER_JOINT_NAMES]
        gripper_pos = float(observation.get("gripper.pos", 0.0))
        teleop.set_joint_pos(joint_pos, gripper_pos)
        return

    if teleop.name != "keyboard_piper_tcp":
        return
    if not hasattr(teleop, "set_tcp_target"):
        return

    ik_step = next((step for step in robot_action_processor.steps if hasattr(step, "kinematics")), None)
    if ik_step is None:
        return

    from lerobot.utils.rotation import Rotation
    from lerobot.robots.piper_follower.robot_kinematic_processor import piper_sdk_joints_to_urdf_joints

    observation = robot.get_observation()
    motor_names = [name for name in ik_step.motor_names if name != "gripper"]
    q_current = piper_sdk_joints_to_urdf_joints(
        [float(observation[f"{name}.pos"]) for name in motor_names]
    )
    tcp_transform = ik_step.kinematics.forward_kinematics(q_current)
    pos = tcp_transform[:3, 3]
    rotvec = Rotation.from_matrix(tcp_transform[:3, :3]).as_rotvec()
    gripper_pos = observation.get("gripper.pos")

    teleop.set_tcp_target(
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(rotvec[0]),
        float(rotvec[1]),
        float(rotvec[2]),
        None if gripper_pos is None else float(gripper_pos),
    )
    logging.info(
        "Initialized keyboard Piper TCP target from current robot pose: "
        "x=%.3f y=%.3f z=%.3f wx=%.3f wy=%.3f wz=%.3f",
        pos[0],
        pos[1],
        pos[2],
        rotvec[0],
        rotvec[1],
        rotvec[2],
    )


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    last_loop_status_t = 0.0
    should_fetch_obs = _teleop_needs_robot_observation(
        display_data, teleop_action_processor, robot_action_processor
    )

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation() if should_fetch_obs else {}

        # Push observation to phone teleop (for FK → EE conversion)
        if hasattr(teleop, "set_observation") and obs:
            teleop.set_observation(obs)

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            if sys.stdout.isatty():
                move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        now = time.monotonic()
        if now - last_loop_status_t >= LOOP_STATUS_INTERVAL_S:
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            if sys.stdout.isatty():
                move_cursor_up(1)
            last_loop_status_t = now

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    sanity_check_bimanual_piper_pair(cfg.robot, cfg.teleop)
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot_action_processor = _configure_processors_for_teleop_robot_pair(
        teleop, robot, robot_action_processor
    )
    should_fetch_obs = _teleop_needs_robot_observation(
        cfg.display_data, teleop_action_processor, robot_action_processor
    ) or _teleop_robot_pair_needs_initial_observation(teleop, robot)
    _configure_robot_for_lightweight_teleop(robot, should_fetch_obs)

    teleop.connect()
    robot.connect()
    _sync_keyboard_piper_tcp_target_from_robot(teleop, robot, robot_action_processor)

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
