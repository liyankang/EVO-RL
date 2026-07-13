#!/usr/bin/env python

from typing import Any

import torch

from lerobot.processor import (
    DeviceProcessorStep,
    ImageCropResizeProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.pipeline import ProcessorStep
from lerobot.values.mlpvalue.configuration_mlpvalue import MlpValueConfig


def make_mlpvalue_pre_post_processors(
    config: MlpValueConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [
            k for k in (config.input_features or {})
            if k.startswith("observation.images")
        ]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
    ]

    if camera_features:
        input_steps.append(
            ImageCropResizeProcessorStep(resize_size=(224, 224))
        )

    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(input_steps),
        PolicyProcessorPipeline(output_steps),
    )
