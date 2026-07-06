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
    """轻量 MLP 价值函数，输入 observation.state + 可选图像，输出标量 value。"""

    # 输入字段
    state_feature: str = OBS_STATE
    task_field: str = "task"
    target_key: str = "observation.value_target"
    loss_weight_key: str = "observation.value_loss_weight"

    # 图像输入（空列表=不使用图像）
    camera_features: list[str] = field(default_factory=list)
    image_encoder: str = "resnet18"   # "resnet18" | "resnet34" | "none"
    freeze_image_encoder: bool = True

    # 网络结构
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.1

    # 分布式 value head
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0

    # 归一化
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    # 运行时
    dtype: str = "float32"

    # 占位
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
