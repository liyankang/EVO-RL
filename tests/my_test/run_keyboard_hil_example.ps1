$env:PYTHONPATH = "src"

# Minimal example for keyboard-driven human-in-the-loop recording on Piper-X follower.
# Replace placeholders before running.

lerobot-human-inloop-record `
  --robot.type=piperx_follower `
  --robot.port=<FOLLOWER_CAN_PORT> `
  --robot.id=my_piperx_follower `
  --robot.require_calibration=false `
  --teleop.type=keyboard_piper_tcp `
  --teleop.id=kb `
  --dataset.repo_id=<HF_USER>/<DATASET_NAME> `
  --dataset.single_task="<YOUR_TASK>" `
  --dataset.num_episodes=1 `
  --dataset.episode_time_s=60 `
  --dataset.reset_time_s=10 `
  --policy.path=<YOUR_POLICY_PATH>
