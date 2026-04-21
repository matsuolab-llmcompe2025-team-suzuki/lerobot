#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_state_sanitizer_processor_step")
@dataclass
class Pi05StateSanitizerProcessorStep(ProcessorStep):
    """Sanitize observation.state by replacing out-of-range values with previous frame values.

    HSR physical joint limits define the valid range. Values outside (sensor errors,
    data corruption) are replaced with the most recent valid value. On the very first
    frame, out-of-range values are clamped to the nearest limit as a fallback.

    This step MUST run BEFORE NormalizerProcessorStep so that normalization receives
    clean values only.
    """

    # HSR joint limits for anomaly detection.
    # Based on URDF (ToyotaResearchInstitute/hsr_description) + data distribution.
    # gripper: URDF says [-0.798, 1.24] but real data has 17% of frames below -0.798
    #          (normal operation). Use data p0.01/p99.99 instead, excluding 45+ corruption.
    # head_pan: URDF says [-3.84, 1.75]. Previous value -1.80 was too strict.
    state_ranges: dict = field(default_factory=lambda: {
        0: (-0.05, 0.70),   # arm_lift (URDF: 0.0~0.69 + margin)
        1: (-2.80, 0.10),   # arm_flex (URDF: -2.62~0.0 + margin)
        2: (-2.10, 3.85),   # arm_roll (URDF: -2.09~3.84 + margin)
        3: (-2.00, 1.30),   # wrist_flex (URDF: -1.92~1.22 + margin)
        4: (-2.00, 3.70),   # wrist_roll (URDF: -1.92~3.67 + margin)
        5: (-1.10, 1.30),   # gripper (data: p0.1=-1.03, p99.9=1.27. URDF range too narrow)
        6: (-4.00, 1.85),   # head_pan (URDF: -3.84~1.75 + margin)
        7: (-1.70, 0.60),   # head_tilt
    })

    _prev_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def reset(self):
        """Clear previous state on episode reset."""
        self._prev_state = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            return transition

        transition = transition.copy()
        state = state.clone()

        for dim, (lo, hi) in self.state_ranges.items():
            if dim >= state.shape[-1]:
                continue
            bad = (state[..., dim] < lo) | (state[..., dim] > hi)
            if not bad.any():
                continue
            if self._prev_state is not None:
                # Replace with previous valid value
                state[..., dim][bad] = self._prev_state[..., dim][bad]
            else:
                # First frame fallback: clamp to nearest limit
                state[..., dim] = state[..., dim].clamp(lo, hi)

        self._prev_state = state.clone().detach()
        transition[TransitionKey.OBSERVATION][OBS_STATE] = state
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # np.linspace は固定値なので 1 回だけ計算してキャッシュ
    _bins: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._bins = np.linspace(-1, 1, 256 + 1)[:-1]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # deepcopy は不要: 元 tensor を変更しないので clone で十分
        state = state.clone()

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=self._bins) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def _load_per_timestamp_action_stats(
    config: PI05Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Optionally override ACTION mean/std with per-timestamp statistics from a sidecar npz file."""
    if dataset_stats is None or not config.use_per_timestamp_action_stats:
        return dataset_stats

    if config.per_timestamp_stats_path is None:
        raise ValueError("per_timestamp_stats_path is required when use_per_timestamp_action_stats=True")

    stats_path = Path(config.per_timestamp_stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Per-timestamp stats file not found: {stats_path}")

    npz = np.load(stats_path)
    mean_hd = npz["mean_hd"]
    std_hd = npz["std_hd"]

    if mean_hd.shape != std_hd.shape:
        raise ValueError(
            f"Per-timestamp stats shape mismatch: mean={mean_hd.shape}, std={std_hd.shape}"
        )

    merged_stats = deepcopy(dataset_stats)
    merged_stats.setdefault(ACTION, {})
    merged_stats[ACTION]["mean"] = torch.as_tensor(mean_hd, dtype=torch.float32)
    merged_stats[ACTION]["std"] = torch.as_tensor(std_hd, dtype=torch.float32)
    return merged_stats


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    dataset_stats = _load_per_timestamp_action_stats(config, dataset_stats)

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # State sanitizer: replace sensor outliers with previous valid values.
        # MUST run BEFORE normalizer so that normalization receives clean values only.
        Pi05StateSanitizerProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalization_clip=config.normalization_clip,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
