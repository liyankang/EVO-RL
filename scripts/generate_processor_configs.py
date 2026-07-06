#!/usr/bin/env python
"""为 openpi 旧格式 checkpoint 直接生成 processor JSON 配置文件（无需网络）。

用法：
  python scripts/generate_processor_configs.py \
    --checkpoint_dir /home/user/openpi05/checkpoints/pi05_piper/toiletButton_0605/29999
"""

import argparse
import json
import sys
from pathlib import Path


PREPROCESSOR_CONFIG = {
    "name": "policy_preprocessor",
    "steps": [
        {
            "registry_name": "rename_observations_processor",
            "config": {"rename_map": {}},
        },
        {
            "registry_name": "to_batch_processor",
        },
        {
            "registry_name": "normalizer_processor",
            "config": {
                "features": {},
                "norm_map": {
                    "ACTION": "IDENTITY",
                    "STATE": "QUANTILES",
                    "VISUAL": "IDENTITY",
                },
                "stats": {},
            },
        },
        {
            "registry_name": "pi05_prepare_state_tokenizer_processor_step",
            "config": {
                "max_state_dim": 32,
                "task_key": "task",
            },
        },
        {
            "registry_name": "tokenizer_processor",
            "config": {
                "tokenizer_name": "google/paligemma-3b-pt-224",
                "max_length": 200,
                "padding_side": "right",
                "padding": "max_length",
            },
        },
        {
            "registry_name": "device_processor",
            "config": {"device": "cuda"},
        },
    ],
}


POSTPROCESSOR_CONFIG = {
    "name": "policy_postprocessor",
    "steps": [
        {
            "registry_name": "unnormalizer_processor",
            "config": {
                "features": {},
                "norm_map": {
                    "ACTION": "IDENTITY",
                    "STATE": "QUANTILES",
                    "VISUAL": "IDENTITY",
                },
                "stats": {},
            },
        },
        {
            "registry_name": "device_processor",
            "config": {"device": "cpu"},
        },
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="openpi checkpoint 路径")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        print(f"错误：目录不存在 {checkpoint_dir}")
        sys.exit(1)

    pre_path = checkpoint_dir / "policy_preprocessor.json"
    post_path = checkpoint_dir / "policy_postprocessor.json"

    with open(pre_path, "w") as f:
        json.dump(PREPROCESSOR_CONFIG, f, indent=2)
    with open(post_path, "w") as f:
        json.dump(POSTPROCESSOR_CONFIG, f, indent=2)

    print(f"已生成:")
    print(f"  {pre_path}")
    print(f"  {post_path}")
    print("\n现在可以重新运行训练命令了。")


if __name__ == "__main__":
    main()
