import numpy as np

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig
from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_SDK_MOTOR_NAMES,
    make_piper_kinematics,
)
from lerobot.utils.rotation import Rotation


PORT = "can0"
ROBOT_ID = "my_piper_follower"
IK_ITERATIONS = 20


def rotation_angle_error_deg(target_rotation: np.ndarray, actual_rotation: np.ndarray) -> float:
    rotation_error = target_rotation.T @ actual_rotation
    cos_angle = (np.trace(rotation_error) - 1.0) / 2.0
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.rad2deg(angle_rad))


def sdk_joints_to_urdf_joints(joints_deg: np.ndarray) -> np.ndarray:
    urdf_joints_deg = joints_deg.copy()
    urdf_joints_deg[1] = -urdf_joints_deg[1]
    urdf_joints_deg[2] = -urdf_joints_deg[2]
    return urdf_joints_deg
USE_PIPERX = False  # PiperX 改成 True


def main():
    robot_cfg = PiperFollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        require_calibration=False,
        enable_on_connect=False,
        can_auto_init=True,
        startup_sleep_s=1.0,
    )
    robot = PiperFollower(robot_cfg)

    robot.connect(calibrate=False)

    try:
        obs = robot.get_observation()

        joints_deg = np.array(
            [float(obs[f"{name}.pos"]) for name in PIPER_SDK_MOTOR_NAMES],
            dtype=float,
        )

        print("Current joints deg:")
        for name, value in zip(PIPER_SDK_MOTOR_NAMES, joints_deg):
            print(f"  {name}: {value:.3f}")

        urdf_joints_deg = sdk_joints_to_urdf_joints(joints_deg)

        print("\nURDF joints deg used for FK/IK:")
        for name, value in zip(PIPER_SDK_MOTOR_NAMES, urdf_joints_deg):
            print(f"  {name}: {value:.3f}")

        kinematics = make_piper_kinematics()
        tcp_T = kinematics.forward_kinematics(urdf_joints_deg)
        tcp_xyz = tcp_T[:3, 3]
        tcp_rotvec = Rotation.from_matrix(tcp_T[:3, :3]).as_rotvec()

        print("\nCurrent TCP:")
        print(f"  x:  {tcp_xyz[0]:.4f} m")
        print(f"  y:  {tcp_xyz[1]:.4f} m")
        print(f"  z:  {tcp_xyz[2]:.4f} m")
        print(f"  wx: {tcp_rotvec[0]:.4f} rad")
        print(f"  wy: {tcp_rotvec[1]:.4f} rad")
        print(f"  wz: {tcp_rotvec[2]:.4f} rad")

        ik_joints_deg = urdf_joints_deg.copy()
        for _ in range(IK_ITERATIONS):
            ik_joints_deg = kinematics.inverse_kinematics(
                current_joint_pos=ik_joints_deg,
                desired_ee_pose=tcp_T,
                position_weight=1.0,
                orientation_weight=1.0,
            )

        print("\nIK joints deg from same TCP:")
        for name, value in zip(PIPER_SDK_MOTOR_NAMES, ik_joints_deg):
            print(f"  {name}: {value:.3f}")

        tcp_T_from_ik = kinematics.forward_kinematics(ik_joints_deg)
        tcp_xyz_from_ik = tcp_T_from_ik[:3, 3]
        tcp_rotvec_from_ik = Rotation.from_matrix(tcp_T_from_ik[:3, :3]).as_rotvec()

        position_error_mm = np.linalg.norm(tcp_xyz_from_ik - tcp_xyz) * 1000.0
        orientation_error_deg = rotation_angle_error_deg(tcp_T[:3, :3], tcp_T_from_ik[:3, :3])

        print("\nFK(IK joints) TCP:")
        print(f"  x:  {tcp_xyz_from_ik[0]:.4f} m")
        print(f"  y:  {tcp_xyz_from_ik[1]:.4f} m")
        print(f"  z:  {tcp_xyz_from_ik[2]:.4f} m")
        print(f"  wx: {tcp_rotvec_from_ik[0]:.4f} rad")
        print(f"  wy: {tcp_rotvec_from_ik[1]:.4f} rad")
        print(f"  wz: {tcp_rotvec_from_ik[2]:.4f} rad")

        print("\nIK verification error:")
        print(f"  position error:    {position_error_mm:.3f} mm")
        print(f"  orientation error: {orientation_error_deg:.3f} deg")

        if position_error_mm < 3.0 and orientation_error_deg < 3.0:
            print("  result: IK reaches the same TCP. Different joints indicate another valid IK solution.")
        else:
            print("  result: IK did not reproduce the TCP closely. Check convergence, URDF, or joint units.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
