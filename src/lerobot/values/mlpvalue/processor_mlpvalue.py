#!/usr/bin/env python

from typing import Any

import torch

from lerobot.processor import (
    DeviceProcessorStep,
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
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(input_steps),
        PolicyProcessorPipeline(output_steps),
    )
