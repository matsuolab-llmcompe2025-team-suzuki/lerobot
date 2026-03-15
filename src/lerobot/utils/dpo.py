"""
dpo.py — DPO-FM (Direct Preference Optimization for Flow Matching)

NORA-1.5 方式の DPO Loss 実装。
flow matching の velocity prediction MSE を log probability の proxy として使用し、
preferred/rejected ペアから行動品質を改善する。

参考: NORA-1.5 (arXiv: 2511.14659), Diffusion-DPO (arXiv: 2311.12908)
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@dataclass
class DPOConfig:
    """DPO-FM の設定"""

    beta: float = 1000.0  # DPO の温度パラメータ（flow matching 向けの大きな値）
    reference_path: str | None = None  # SFT チェックポイントのパス
    alpha: float = 1.0  # DPO/SFT 配分: 1.0=純粋DPO, 0.5=Hybrid, 0.0=純粋SFT
    dim_weights: list[float] | None = None  # action 次元別重み（None=均等）


def policy_forward_with_noise(
    policy,
    batch: dict[str, Tensor],
    noise: Tensor,
    time: Tensor,
    dim_weights: Tensor | None = None,
) -> Tensor:
    """
    Pi0_5Policy の内部処理を行い、指定された noise/time で flow matching loss を計算する。

    preferred/rejected で同一の noise/time を共有するために、
    policy.forward() を直接呼ばず、内部の Pi05FlowMatching.forward() に
    noise/time を渡す。

    Args:
        policy: Pi0_5Policy（学習モデル or reference モデル）
        batch: preprocessor 適用済みの batch dict
        noise: 共有ノイズ (B, chunk_size, max_action_dim)
        time: 共有タイムステップ (B,)
        dim_weights: action 次元別重み (action_dim,) — None なら均等

    Returns:
        per_sample_mse: (B,) — 各サンプルの（重み付き）平均 MSE
    """
    # Pi0_5Policy の前処理（画像・トークン・アクション）
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    # Pi05FlowMatching.forward() に noise/time を明示的に渡す
    # 返り値: (B, chunk_size, max_action_dim) の MSE（reduction="none"）
    losses = policy.model.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)

    # 実際の action_dim に切り詰め
    original_action_dim = policy.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :original_action_dim]

    # Per-sample に集約: (B, chunk_size, action_dim) → (B,)
    if dim_weights is not None:
        # 次元別重み付き平均（例: グリッパ次元を重視）
        # dim_weights: (action_dim,) → broadcast to (1, 1, action_dim)
        losses = losses * dim_weights.unsqueeze(0).unsqueeze(0)
        per_sample_mse = losses.sum(dim=2).mean(dim=1) / dim_weights.sum()
    else:
        per_sample_mse = losses.mean(dim=(1, 2))

    return per_sample_mse


def compute_dpo_fm_loss(
    policy,
    ref_policy,
    batch_preferred: dict[str, Tensor],
    batch_rejected: dict[str, Tensor],
    dpo_config: DPOConfig,
) -> tuple[Tensor, dict]:
    """
    NORA-1.5 方式の DPO-FM Loss を計算する。
    dpo_config.alpha で Hybrid SFT+DPO にも対応。

    DPO-FM Loss:
        L = -E[log σ(-β * ((MSE_θ(W) - MSE_ref(W)) - (MSE_θ(L) - MSE_ref(L))))]

    Hybrid (alpha < 1.0):
        L_total = alpha * L_DPO + (1 - alpha) * L_SFT

    Args:
        policy: 学習中のモデル (θ)
        ref_policy: 凍結した reference model (θ_ref)
        batch_preferred: preferred (winner) batch
        batch_rejected: rejected (loser) batch
        dpo_config: DPO 設定

    Returns:
        loss: DPO-FM loss (scalar)
        metrics: WandB 用メトリクス dict
    """
    beta = dpo_config.beta

    # dim_weights の準備
    dim_weights_tensor = None
    if dpo_config.dim_weights is not None:
        dim_weights_tensor = torch.tensor(dpo_config.dim_weights, dtype=torch.float32)
        # policy と同じデバイスに移動
        actions_for_device = policy.prepare_action(batch_preferred)
        dim_weights_tensor = dim_weights_tensor.to(actions_for_device.device)

    # 同一 noise/time を preferred/rejected で共有
    actions_pref = policy.prepare_action(batch_preferred)
    noise = policy.model.sample_noise(actions_pref.shape, actions_pref.device)
    time = policy.model.sample_time(actions_pref.shape[0], actions_pref.device)

    # 学習モデルの MSE
    mse_pref = policy_forward_with_noise(policy, batch_preferred, noise, time, dim_weights_tensor)
    mse_rej = policy_forward_with_noise(policy, batch_rejected, noise, time, dim_weights_tensor)

    # Reference モデルの MSE
    with torch.no_grad():
        ref_mse_pref = policy_forward_with_noise(ref_policy, batch_preferred, noise, time, dim_weights_tensor)
        ref_mse_rej = policy_forward_with_noise(ref_policy, batch_rejected, noise, time, dim_weights_tensor)

    # DPO-FM loss
    # MSE が小さい = 尤度が高い → 符号を反転して logits を計算
    logits = -beta * ((mse_pref - ref_mse_pref) - (mse_rej - ref_mse_rej))
    dpo_loss = -F.logsigmoid(logits).mean()

    # Hybrid SFT+DPO: alpha < 1.0 の場合、SFT loss をブレンド
    # alpha=1.0 → 純粋 DPO, alpha=0.5 → Hybrid, alpha=0.0 → 純粋 SFT
    if dpo_config.alpha < 1.0:
        sft_loss, _ = policy.forward(batch_preferred)
        loss = dpo_config.alpha * dpo_loss + (1.0 - dpo_config.alpha) * sft_loss
    else:
        sft_loss = None
        loss = dpo_loss

    # WandB 用メトリクス
    with torch.no_grad():
        metrics = {
            "dpo/loss": dpo_loss.item(),
            "dpo/total_loss": loss.item(),
            "dpo/preferred_mse": mse_pref.mean().item(),
            "dpo/rejected_mse": mse_rej.mean().item(),
            "dpo/accuracy": (logits > 0).float().mean().item(),
            "dpo/reward_margin": (mse_rej - mse_pref).mean().item(),
            "dpo/ref_preferred_mse": ref_mse_pref.mean().item(),
            "dpo/ref_rejected_mse": ref_mse_rej.mean().item(),
            "dpo/alpha": dpo_config.alpha,
        }
        if sft_loss is not None:
            metrics["dpo/sft_loss"] = sft_loss.item()

    return loss, metrics
