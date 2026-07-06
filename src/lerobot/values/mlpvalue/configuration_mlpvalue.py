#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("mlpvalue")
@dataclass
class MlpValueConfig(PreTrainedConfig):
    """轻量 MLP 价值函数，输入 observation.state，输出标量 value。"""

    # 输入字段
    state_feature: str = OBS_STATE
    target_key: str = "observation.value_target"
    loss_weight_key: str = "observation.value_loss_weight"

    # 网络结构
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.1

    # 分布式 value head（与 pistar06 保持一致，便于直接替换）
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0

    # 归一化
    normalization_mode: NormalizationMode = NormalizationMode.MEAN_STD

    # 运行时
    dtype: str = "float32"

    # 占位（make_policy 通用接口需要）
    push_to_hub: bool = False
    repo_id: str | None = None

    def validate(self) -> None:
        if not self.hidden_dims:
            raise ValueError("'value.hidden_dims' must be non-empty.")
        if self.num_bins < 2:
            raise ValueError("'value.num_bins' must be >= 2.")

    def validate_features(self) -> None:
        return

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=1e-4,
            decay_lr=1e-6,
            num_warmup_steps=500,
            num_decay_steps=7500,
        )
