#!/usr/bin/env python

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as tv_models
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.values.mlpvalue.configuration_mlpvalue import MlpValueConfig
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_normalized_value_targets,
    expected_value_from_logits,
    project_values_to_bins,
    resolve_episode_success_label,
)
from lerobot.utils.recording_annotations import EPISODE_SUCCESS

# ResNet 输出特征维度
_RESNET_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

# ImageNet 归一化参数（用于图像预处理）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _build_resnet(name: str, freeze: bool) -> nn.Module:
    weights = {"resnet18": tv_models.ResNet18_Weights.DEFAULT,
               "resnet34": tv_models.ResNet34_Weights.DEFAULT,
               "resnet50": tv_models.ResNet50_Weights.DEFAULT}[name]
    model = getattr(tv_models, name)(weights=weights)
    model.fc = nn.Identity()   # 去掉分类头，输出 512-d 特征
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


class MlpValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_bins: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MlpValuePolicy(PreTrainedPolicy):
    config_class = MlpValueConfig
    name = "mlpvalue"

    def __init__(self, config: MlpValueConfig, dataset_meta=None, **kwargs):
        super().__init__(config)
        self.config = config

        dataset_stats = kwargs.pop("dataset_stats", None)
        stats = dataset_meta.stats if (dataset_meta is not None and hasattr(dataset_meta, "stats")) else dataset_stats

        state_dim = self._infer_state_dim(stats, dataset_meta)

        # 图像编码器
        self.camera_features = list(config.camera_features)
        self.image_encoder = None
        img_feat_dim = 0
        if self.camera_features and config.image_encoder != "none":
            self.image_encoder = _build_resnet(config.image_encoder, config.freeze_image_encoder)
            img_feat_dim = _RESNET_FEAT_DIM[config.image_encoder] * len(self.camera_features)
            # ImageNet 归一化参数注册为 buffer（随设备移动）
            self.register_buffer("img_mean", _IMAGENET_MEAN.clone())
            self.register_buffer("img_std",  _IMAGENET_STD.clone())

        input_dim = state_dim + img_feat_dim
        self.model = MlpValueNetwork(
            input_dim=input_dim,
            hidden_dims=list(config.hidden_dims),
            num_bins=config.num_bins,
            dropout=config.dropout,
        )

        bin_centers = torch.linspace(config.bin_min, config.bin_max, config.num_bins)
        self.register_buffer("bin_centers", bin_centers)

    def _infer_state_dim(self, dataset_stats, dataset_meta) -> int:
        key = self.config.state_feature
        if self.config.input_features:
            feat = self.config.input_features.get(key)
            if feat is not None:
                shape = feat.shape if hasattr(feat, "shape") else feat.get("shape", None)
                if shape is not None:
                    return int(np.prod(shape))
        if dataset_meta is not None and hasattr(dataset_meta, "features"):
            feat = dataset_meta.features.get(key)
            if feat is not None:
                shape = feat.shape if hasattr(feat, "shape") else feat.get("shape", None)
                if shape is not None:
                    return int(np.prod(shape))
        if dataset_stats and key in dataset_stats:
            mean = dataset_stats[key].get("mean")
            if mean is not None:
                return int(mean.numel()) if isinstance(mean, Tensor) else len(mean)
        raise ValueError(f"Cannot infer state_dim for '{key}'. Pass dataset_meta or dataset_stats.")

    def _encode_images(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.camera_features or self.image_encoder is None:
            return None
        feats = []
        for cam in self.camera_features:
            img = batch[cam]  # (B, C, H, W) float in [0,1] or uint8
            if not isinstance(img, Tensor):
                img = torch.as_tensor(img)
            img = img.to(device=next(self.parameters()).device, dtype=torch.float32)
            if img.max() > 1.5:   # uint8 [0,255] → [0,1]
                img = img / 255.0
            img = (img - self.img_mean) / self.img_std
            feats.append(self.image_encoder(img))
        return torch.cat(feats, dim=-1)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        return

    def predict_action_chunk(self, batch, **kwargs):
        raise RuntimeError("MlpValuePolicy is a value model and does not support action prediction.")

    def select_action(self, batch, **kwargs):
        raise RuntimeError("MlpValuePolicy is a value model and does not support action selection.")

    def _build_input(self, batch: dict[str, Tensor]) -> Tensor:
        device = next(self.parameters()).device
        state = batch[self.config.state_feature]
        if not isinstance(state, Tensor):
            state = torch.as_tensor(state)
        state = state.to(device=device, dtype=torch.float32, non_blocking=True)

        img_feat = self._encode_images(batch)
        if img_feat is not None:
            img_feat = img_feat.to(device=device)
            return torch.cat([state, img_feat], dim=-1)
        return state

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.target_key not in batch:
            raise KeyError(f"Missing target key '{self.config.target_key}' in batch.")

        device = next(self.parameters()).device
        x = self._build_input(batch)

        value_target = batch[self.config.target_key]
        if not isinstance(value_target, Tensor):
            value_target = torch.as_tensor(value_target)
        value_target = value_target.to(device=device, dtype=torch.float32, non_blocking=True)
        if value_target.ndim == 2 and value_target.shape[-1] == 1:
            value_target = value_target.squeeze(-1)

        logits = self.model(x)
        bin_centers = self.bin_centers.to(device=device)
        soft_target = project_values_to_bins(value_target, bin_centers)
        log_probs = functional.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

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
        x = self._build_input(batch)
        logits = self.model(x)
        return expected_value_from_logits(logits, self.bin_centers.to(logits.device))

    def build_training_raw_batch_hook(self, dataset, targets_cfg):
        raw_frames = dataset.hf_dataset.with_format(None)
        if len(raw_frames) == 0:
            raise ValueError("Dataset has no frames.")

        episode_indices  = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices    = np.asarray(raw_frames["frame_index"],   dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"],         dtype=np.int64)

        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes    = episodes_ds[:]
        n_episodes  = len(episodes_ds)
        has_success = targets_cfg.success_field in episodes_ds.column_names

        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}
        for i in range(n_episodes):
            ep_idx    = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks     = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved = resolve_episode_success_label(explicit_success, default_label=targets_cfg.default_success, require_label=True)
            ep_success = resolved == EPISODE_SUCCESS

            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx, task_index=task_index, length=ep_length, success=ep_success
            )
            task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices, frame_indices=frame_indices,
            episode_info=episode_info, task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=self.config.bin_min, clip_max=self.config.bin_max,
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
                raise KeyError("Missing 'index' in batch.")
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            idx_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            batch[target_key] = torch.from_numpy(value_target_lookup[idx_np]).to(dtype=torch.float32)
            return batch

        return value_target_hook
