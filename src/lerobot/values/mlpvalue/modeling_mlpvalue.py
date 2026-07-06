#!/usr/bin/env python

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.values.mlpvalue.configuration_mlpvalue import MlpValueConfig
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_normalized_value_targets,
    expected_value_from_logits,
    project_values_to_bins,
    resolve_episode_success_label,
)
from lerobot.utils.recording_annotations import EPISODE_SUCCESS


class MlpValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: list[int], num_bins: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class MlpValuePolicy(PreTrainedPolicy):
    config_class = MlpValueConfig
    name = "mlpvalue"

    def __init__(self, config: MlpValueConfig, dataset_meta=None, **kwargs):
        super().__init__(config)
        self.config = config

        # 从 dataset_meta 或 dataset_stats（兼容两种调用方式）推断 state_dim
        dataset_stats = kwargs.pop("dataset_stats", None)
        if dataset_meta is not None and hasattr(dataset_meta, "stats"):
            stats = dataset_meta.stats
        else:
            stats = dataset_stats

        state_dim = self._infer_state_dim(stats, dataset_meta)
        self.model = MlpValueNetwork(
            state_dim=state_dim,
            hidden_dims=list(config.hidden_dims),
            num_bins=config.num_bins,
            dropout=config.dropout,
        )

        bin_centers = torch.linspace(config.bin_min, config.bin_max, config.num_bins)
        self.register_buffer("bin_centers", bin_centers)

    def _infer_state_dim(self, dataset_stats: dict | None, dataset_meta=None) -> int:
        # 优先从 input_features 拿（保存的 config 里有）
        if self.config.input_features:
            feat = self.config.input_features.get(self.config.state_feature)
            if feat is not None:
                shape = feat.get("shape", None)
                if shape:
                    return int(np.prod(shape))
        # 从 dataset_meta.features 拿
        if dataset_meta is not None and hasattr(dataset_meta, "features"):
            feat = dataset_meta.features.get(self.config.state_feature)
            if feat is not None:
                shape = feat.get("shape", None)
                if shape:
                    return int(np.prod(shape))
        # 从 stats 拿
        if dataset_stats and self.config.state_feature in dataset_stats:
            mean = dataset_stats[self.config.state_feature].get("mean")
            if mean is not None:
                if isinstance(mean, Tensor):
                    return int(mean.numel())
                return len(mean)
        raise ValueError(
            f"Cannot infer state_dim for '{self.config.state_feature}'. "
            "Pass dataset_meta or dataset_stats."
        )

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        return

    def predict_action_chunk(self, batch, **kwargs):
        raise RuntimeError("MlpValuePolicy is a value model and does not support action prediction.")

    def select_action(self, batch, **kwargs):
        raise RuntimeError("MlpValuePolicy is a value model and does not support action selection.")

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.target_key not in batch:
            raise KeyError(
                f"Missing target key '{self.config.target_key}' in batch. "
                "Make sure lerobot-value-train target hook is enabled."
            )

        state = batch[self.config.state_feature]
        if not isinstance(state, Tensor):
            state = torch.as_tensor(state)
        device = next(self.parameters()).device
        state = state.to(device=device, dtype=torch.float32, non_blocking=True)

        value_target = batch[self.config.target_key]
        if not isinstance(value_target, Tensor):
            value_target = torch.as_tensor(value_target)
        value_target = value_target.to(device=device, dtype=torch.float32, non_blocking=True)
        if value_target.ndim == 2 and value_target.shape[-1] == 1:
            value_target = value_target.squeeze(-1)

        logits = self.model(state)
        bin_centers = self.bin_centers.to(device=device)
        soft_target = project_values_to_bins(value_target, bin_centers)
        log_probs = functional.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

        # 可选 loss weight
        if self.config.loss_weight_key in batch:
            w = batch[self.config.loss_weight_key]
            if not isinstance(w, Tensor):
                w = torch.as_tensor(w)
            w = w.to(device=device, dtype=torch.float32, non_blocking=True)
            if w.ndim == 2 and w.shape[-1] == 1:
                w = w.squeeze(-1)
            per_sample_loss = per_sample_loss * w

        loss = per_sample_loss.mean() if reduction == "mean" else per_sample_loss.sum()

        with torch.no_grad():
            pred_value = expected_value_from_logits(logits, bin_centers)
            mae = (pred_value - value_target).abs().mean()

        return loss, {"loss": loss.item(), "mae": mae.item()}

    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[self.config.state_feature]
        if not isinstance(state, Tensor):
            state = torch.as_tensor(state)
        state = state.to(device=next(self.parameters()).device, dtype=torch.float32)
        logits = self.model(state)
        bin_centers = self.bin_centers.to(device=logits.device)
        return expected_value_from_logits(logits, bin_centers)

    def build_training_raw_batch_hook(self, dataset, targets_cfg):
        """与 pistar06 完全相同的 hook 逻辑，复用 compute_normalized_value_targets。"""
        raw_frames = dataset.hf_dataset.with_format(None)
        if len(raw_frames) == 0:
            raise ValueError("Dataset has no frames.")

        episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes = episodes_ds[:]
        n_episodes = len(episodes_ds)
        has_success = targets_cfg.success_field in episodes_ds.column_names

        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}
        for i in range(n_episodes):
            ep_idx = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved_success = resolve_episode_success_label(
                explicit_success,
                default_label=targets_cfg.default_success,
                require_label=True,
            )
            ep_success = resolved_success == EPISODE_SUCCESS

            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx,
                task_index=task_index,
                length=ep_length,
                success=ep_success,
            )
            task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=self.config.bin_min,
            clip_max=self.config.bin_max,
            mid_bonus_scale=targets_cfg.mid_bonus_scale,
            mid_bonus_sigma=targets_cfg.mid_bonus_sigma,
        )

        max_index = int(np.max(absolute_indices))
        value_target_lookup = np.zeros(max_index + 1, dtype=np.float32)
        value_target_lookup[absolute_indices] = value_targets.astype(np.float32, copy=False)

        target_key = targets_cfg.target_field

        def value_target_hook(batch: dict[str, Any], step: int) -> dict[str, Any]:
            del step
            batch_indices = batch.get("index")
            if batch_indices is None:
                raise KeyError("Missing 'index' in batch while building value targets.")
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            idx_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            batch[target_key] = torch.from_numpy(value_target_lookup[idx_np]).to(dtype=torch.float32)
            return batch

        return value_target_hook
