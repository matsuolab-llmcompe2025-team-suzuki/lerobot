"""
dpo.py — DPO-FM (Direct Preference Optimization for Flow Matching)

NORA-1.5 方式の DPO Loss 実装。
flow matching の velocity prediction MSE を log probability の proxy として使用し、
preferred/rejected ペアから行動品質を改善する。

参考: NORA-1.5 (arXiv: 2511.14659), Diffusion-DPO (arXiv: 2311.12908)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@dataclass
class DPOConfig:
    """DPO-FM の設定"""

    beta: float = 1000.0  # DPO の温度パラメータ（flow matching 向けの大きな値）
    reference_path: str | None = None  # SFT チェックポイントのパス


def policy_forward_with_noise(
    policy,
    batch: dict[str, Tensor],
    noise: Tensor,
    time: Tensor,
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

    Returns:
        per_sample_mse: (B,) — 各サンプルの平均 MSE
    """
    # Pi0_5Policy の前処理（画像・トークン・アクション）
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    # Pi05FlowMatching.forward() に noise/time を明示的に渡す
    # 返り値: (B, chunk_size, action_dim) の MSE（reduction="none"）
    losses = policy.model.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)

    # 実際の action_dim に切り詰め
    original_action_dim = policy.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :original_action_dim]

    # Per-sample に集約: (B, chunk_size, action_dim) → (B,)
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

    DPO-FM Loss:
        L = -E[log σ(-β * ((MSE_θ(W) - MSE_ref(W)) - (MSE_θ(L) - MSE_ref(L))))]

    MSE が小さい = velocity prediction が正確 = 尤度が高い
    → preferred の MSE を下げ、rejected の MSE を上げるように学習

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

    # 同一 noise/time を preferred/rejected で共有
    # preferred の action shape を基準にノイズを生成
    actions_pref = policy.prepare_action(batch_preferred)
    noise = policy.model.sample_noise(actions_pref.shape, actions_pref.device)
    time = policy.model.sample_time(actions_pref.shape[0], actions_pref.device)

    # 学習モデルの MSE
    mse_pref = policy_forward_with_noise(policy, batch_preferred, noise, time)  # (B,)
    mse_rej = policy_forward_with_noise(policy, batch_rejected, noise, time)  # (B,)

    # Reference モデルの MSE
    with torch.no_grad():
        ref_mse_pref = policy_forward_with_noise(ref_policy, batch_preferred, noise, time)  # (B,)
        ref_mse_rej = policy_forward_with_noise(ref_policy, batch_rejected, noise, time)  # (B,)

    # DPO-FM loss
    # MSE が小さい = 尤度が高い → 符号を反転して logits を計算
    logits = -beta * ((mse_pref - ref_mse_pref) - (mse_rej - ref_mse_rej))
    loss = -F.logsigmoid(logits).mean()

    # WandB 用メトリクス
    with torch.no_grad():
        metrics = {
            "dpo/loss": loss.item(),
            "dpo/preferred_mse": mse_pref.mean().item(),
            "dpo/rejected_mse": mse_rej.mean().item(),
            "dpo/accuracy": (logits > 0).float().mean().item(),
            "dpo/reward_margin": (mse_rej - mse_pref).mean().item(),
            "dpo/ref_preferred_mse": ref_mse_pref.mean().item(),
            "dpo/ref_rejected_mse": ref_mse_rej.mean().item(),
        }

    return loss, metrics
