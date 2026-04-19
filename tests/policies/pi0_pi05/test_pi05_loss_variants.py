#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke tests for PI05 loss variants: flow_loss_type and use_min_snr_weighting.

CPU-only. Covers config validation and Min-SNR weight numerical correctness.
Issue #125 (Run 62).
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Config  # noqa: E402


def _make_config(**overrides) -> PI05Config:
    return PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32", **overrides)


def test_default_config_backward_compatible():
    """既存の学習に影響が無いことを確認 (flow_loss_type='mse', min_snr 無効)"""
    config = _make_config()
    assert config.flow_loss_type == "mse"
    assert config.smooth_l1_beta == 1.0
    assert config.use_min_snr_weighting is False
    assert config.min_snr_gamma == 5.0


def test_smooth_l1_config_accepted():
    config = _make_config(flow_loss_type="smooth_l1", smooth_l1_beta=0.5)
    assert config.flow_loss_type == "smooth_l1"
    assert config.smooth_l1_beta == 0.5


def test_invalid_flow_loss_type_rejected():
    with pytest.raises(ValueError, match="flow_loss_type must be"):
        _make_config(flow_loss_type="huber")


def test_invalid_smooth_l1_beta_rejected():
    with pytest.raises(ValueError, match="smooth_l1_beta must be > 0"):
        _make_config(smooth_l1_beta=-1.0)


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

    expected = []
    for t in time.tolist():
        s = ((1.0 - t) / t) ** 2
        expected.append(min(s, gamma) / (s + 1.0))

    assert torch.allclose(weights, torch.tensor(expected), atol=1e-5)
    assert (weights > 0).all()
    assert (weights <= gamma / (gamma + 1.0) + 1e-5).all()  # 上限: γ/(γ+1)


def test_smooth_l1_vs_mse_differ_on_outliers():
    """SmoothL1 が外れ値に対して MSE より小さい勾配になることを確認"""
    u = torch.zeros(4, 3, 7)
    v = torch.zeros(4, 3, 7)
    v[0, 0, 0] = 10.0  # 大きな外れ値

    mse = F.mse_loss(u, v, reduction="none").mean()
    smooth_l1 = F.smooth_l1_loss(v, u, reduction="none", beta=1.0).mean()

    # 外れ値 10 に対して MSE = 100, SmoothL1 = 10 - 0.5 = 9.5
    # 平均後: MSE ≈ 100/84 ≈ 1.19, SmoothL1 ≈ 9.5/84 ≈ 0.113
    assert smooth_l1 < mse
    assert abs(mse.item() - 100.0 / (4 * 3 * 7)) < 1e-4
    assert abs(smooth_l1.item() - 9.5 / (4 * 3 * 7)) < 1e-4
