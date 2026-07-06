"""Retroactively add episode_success='success' to all episodes in a LeRobot v3 dataset."""
import argparse
from pathlib import Path

import pandas as pd


def patch(dataset_path: str, label: str = "success"):
    episodes_dir = Path(dataset_path) / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet found under {episodes_dir}")

    for p in parquet_files:
        df = pd.read_parquet(p)
        if "episode_success" in df.columns:
            print(f"{p}: already has episode_success, overwriting")
        df["episode_success"] = label
        df.to_parquet(p, index=False)
        print(f"{p}: patched {len(df)} episodes → episode_success='{label}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path")
    ap.add_argument("--label", default="success", choices=["success", "failure"])
    args = ap.parse_args()
    patch(args.dataset_path, args.label)
