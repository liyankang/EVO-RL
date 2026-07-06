from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files="/home/user/.cache/huggingface/lerobot/lerobot_v3/data_toiletButton_0609/data/**/*.parquet"
)["train"]

print(ds.column_names)

import numpy as np

adv = np.array(ds["complementary_info.advantage_push_button"])
print(f"count : {len(adv)}")
print(f"mean  : {adv.mean():.4f}")
print(f"std   : {adv.std():.4f}")
print(f"min   : {adv.min():.4f}")
print(f"25%   : {np.percentile(adv, 25):.4f}")
print(f"50%   : {np.percentile(adv, 50):.4f}")
print(f"75%   : {np.percentile(adv, 75):.4f}")
print(f"max   : {adv.max():.4f}")
print(f"pos%  : {(adv > 0).mean()*100:.1f}%  (advantage > 0)")
print(f"neg%  : {(adv < 0).mean()*100:.1f}%  (advantage < 0)")
