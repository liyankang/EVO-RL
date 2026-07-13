import pandas as pd
META = "/home/user/.cache/huggingface/lerobot/lerobot_v3/data_toiletButton_0703/meta/episodes/chunk-000/file-000.parquet"

df = pd.read_parquet(META)

print(df.tail()[[
    "episode_index",
    "dataset_from_index",
    "dataset_to_index",
    "length"
]])