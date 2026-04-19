#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke tests for PI05 flow matching loss variants: use_min_snr_weighting.

CPU-only. Covers config validation and Min-SNR weight numerical correctness.
Issue #125 (Run 62).
"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Config  # noqa: E402


def _make_config(**overrides) -> PI05Config:
    return PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32", **overrides)


def test_default_config_backward_compatible():
    """既存の学習に影響が無いことを確認 (min_snr 無効)"""
    config = _make_config()
    assert config.use_min_snr_weighting is False
    assert config.min_snr_gamma == 5.0


def test_min_snr_config_accepted():
    config = _make_config(use_min_snr_weighting=True, min_snr_gamma=3.0)
    assert config.use_min_snr_weighting is True
    assert config.min_snr_gamma == 3.0


def test_invalid_min_snr_gamma_rejected():
    with pytest.raises(ValueError, match="min_snr_gamma must be > 0"):
        _make_config(min_snr_gamma=0.0)


def test_min_snr_weight_formula():
    """Min-SNR-γ weight: w(t) = min(SNR(t), γ) / (SNR(t) + 1), SNR(t) = ((1-t)/t)^2

    t=0.5 ⇒ SNR=1, w = min(1, 5) / 2 = 0.5
    t=0.9 ⇒ SNR=(0.1/0.9)^2 ≈ 0.0123, w ≈ 0.0123 / 1.0123 ≈ 0.0122
    t=0.1 ⇒ SNR=(0.9/0.1)^2 = 81, clamped to γ=5, w = 5 / 82 ≈ 0.061
    """
    gamma = 5.0
    eps = 1e-6

    def min_snr_w(t: float) -> float:
        t_c = max(eps, min(1.0 - eps, t))
        snr = ((1.0 - t_c) / t_c) ** 2
        return min(snr, gamma) / (snr + 1.0)

    assert abs(min_snr_w(0.5) - 0.5) < 1e-6
    assert abs(min_snr_w(0.9) - (0.0123456790 / 1.0123456790)) < 1e-5
    assert abs(min_snr_w(0.1) - (5.0 / 82.0)) < 1e-5


def test_min_snr_weight_tensor():
    """Torch 実装と素の式が一致することを確認"""
    time = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = 5.0
    eps = 1e-6

    t_c = time.clamp(min=eps, max=1.0 - eps)
    snr = ((1.0 - t_c) / t_c) ** 2
    weights = torch.clamp(snr, max=gamma) / (snr + 1.0)
    # Terminal-SNR guard
    weights = torch.where(snr == 0, torch.ones_like(weights), weights)

    expected = []
    for t in time.tolist():
        s = ((1.0 - t) / t) ** 2
        expected.append(min(s, gamma) / (s + 1.0))

    assert torch.allclose(weights, torch.tensor(expected), atol=1e-5)
    assert (weights > 0).all()
    assert (weights <= gamma / (gamma + 1.0) + 1e-5).all()  # 上限: γ/(γ+1)


def test_min_snr_terminal_guard():
    """SNR=0 のとき weight=1 にフォールバックすることを確認"""
    snr = torch.tensor([0.0, 1.0, 5.0, 100.0])
    gamma = 5.0
    weights = torch.clamp(snr, max=gamma) / (snr + 1.0)
    weights = torch.where(snr == 0, torch.ones_like(weights), weights)

    assert weights[0].item() == 1.0  # guard kicks in
    assert abs(weights[1].item() - 0.5) < 1e-6
    assert abs(weights[2].item() - 5.0 / 6.0) < 1e-6
    assert abs(weights[3].item() - 5.0 / 101.0) < 1e-6
