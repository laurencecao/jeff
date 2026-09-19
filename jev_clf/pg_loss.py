"""Policy-gradient loss for the RLCD Arm A (correctness reward + KL anchor).

Objective. For each training state the policy emits ONE label token; the reward is
1 if that label is the gold label, else 0. Optimised with a policy gradient:

    L = -E[ (R - b) * log pi(a|s) ]  +  beta * KL(pi(.|s) || pi_ref(.|s))

Relationship to cross-entropy (stated because it is easy to get wrong):
correctness-PG maximises `p_y` for the emitted label; CE maximises `log p_y`.
They share the same one-hot optimum on deterministic labels but are different
objectives with different gradients. Neither is, on its own, a proper scoring
rule over SAMPLED outcomes -- this corpus has no sampled outcomes (every label is
single and deterministic). CE IS a strictly proper scoring rule over the label
distribution, which is why CE remains the calibration-safe default and why this
arm must be measured against a matched CE control rather than assumed better.

Why the KL is computed EXACTLY over the whole label distribution, not from the
emitted action. A sampled-action estimator such as `exp(r) - r - 1` is only valid
under the sampling measure it was derived for (a sample from the REFERENCE
policy). Sampling from the current policy and reusing that form gives an invalid
KL. Worse, caching a scalar "reference log-prob of the emitted action" is unusable
here anyway, because the emitted action changes across epochs -- the cache would
be stale exactly when it matters.

The label space is tiny (3 for verdict, up to 5 for score), so there is no reason
to approximate: cache the reference's FULL categorical log-prob vector per state
once, then compute

    KL = sum_a pi(a|s) * (log pi(a|s) - log pi_ref(a|s))

exactly, every step, for any action the policy happens to emit. Cost is O(L) with
L <= 5.

Testability: the loss is a pure function of tensors -- no model, tokenizer, or
device -- so its convergence behaviour is checked on a tiny CPU toy before any GPU
time is spent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PGConfig:
    """Knobs for the policy-gradient arm."""

    # Weight on the exact KL anchor to the frozen reference policy.
    kl_beta: float = 0.05
    # 'mean' | 'none'. 'mean' subtracts the batch mean reward: the cheap variance
    # reduction. A learned value head is out of scope here.
    baseline: str = "mean"
    # Advantage clipping, so one lucky batch cannot take a huge step.
    adv_clip: float | None = 1.0
    # Entropy bonus. Defaulted OFF deliberately: on a 3-way label space an entropy
    # bonus pushes toward the same over-confidence the task is trying to avoid, so
    # enabling it must be an explicit choice rather than a silent default.
    entropy_beta: float = 0.0


def exact_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    atol: float = 1e-4,
) -> torch.Tensor:
    """Exact KL(pi || pi_ref) over the label space. Both inputs are [n, L] LOG-probs.

    Computed over the full support, so it is correct regardless of which action the
    policy emitted. Returns a scalar mean over the batch.

    Both inputs must be VALID normalized log-distributions (exp sums to 1 per row).
    This is asserted, not assumed: silently passing unnormalized tensors -- e.g. raw
    logits, or logprobs that a parameter update has drifted -- yields a number that
    is not a KL divergence at all.
    """
    if policy_logprobs.shape != ref_logprobs.shape:
        raise ValueError("policy and reference log-probs must have the same shape")
    if policy_logprobs.ndim != 2:
        raise ValueError("expected [n, L] log-prob tensors")
    for name, t in (("policy_logprobs", policy_logprobs), ("ref_logprobs", ref_logprobs)):
        sums = t.exp().sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=atol):
            worst = float((sums - 1.0).abs().max())
            raise ValueError(
                f"{name} rows do not sum to 1 (max deviation {worst:.4g}); "
                "pass log_softmax(logits), not raw logits or drifted tensors"
            )
    pi = policy_logprobs.exp()
    return (pi * (policy_logprobs - ref_logprobs)).sum(dim=-1).mean()


def sample_actions(policy_logprobs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw actions a ~ Categorical(pi(.|s)). REQUIRED by the arm.

    REINFORCE's estimator is unbiased only for actions sampled from the policy.
    Taking argmax instead gives a different (biased) quantity and would make the
    arm a mislabeled surrogate rather than a policy-gradient method -- so the
    training loop must not pass argmax here.
    """
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    return torch.distributions.Categorical(logits=policy_logprobs).sample()


def policy_gradient_loss(
    policy_logprobs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    cfg: PGConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Correctness policy gradient with an exact KL anchor.

    Args:
        policy_logprobs: [n, L] log pi(.|s) over the label space.
        actions:         [n] index of the EMITTED label, drawn by sample_actions()
                         as a ~ Categorical(pi). Sampling must be treated as
                         non-differentiable; the gradient flows through
                         log pi(a|s) gathered below. Do NOT pass argmax.
        rewards:         [n] reward for that action (1.0 correct, 0.0 wrong),
                         computed against the gold label for the same state.
        ref_logprobs:    [n, L] frozen reference log pi_ref(.|s), same label order.
        cfg:             PGConfig.

    Returns (loss, diagnostics). Diagnostics are what make the arm auditable: a
    policy-gradient run reporting only a falling loss hides whether it is learning
    anything, so advantage magnitude, exact KL and the reward spread are surfaced.
    """
    cfg = cfg or PGConfig()
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    n, L = policy_logprobs.shape
    if actions.shape != (n,) or rewards.shape != (n,):
        raise ValueError(f"actions and rewards must be [{n}]")
    if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= L):
        raise ValueError("action index out of range for the label space")
    if actions.requires_grad:
        # Sampling is nondifferentiable by construction; a grad-carrying action
        # tensor means the caller derived it from the policy by some other route.
        raise ValueError("actions must not carry grad; sample them and detach")

    # log pi(a|s) for the emitted action, gathered from the full distribution.
    chosen_logprob = policy_logprobs.gather(1, actions.view(-1, 1)).squeeze(1)

    # --- advantage ---------------------------------------------------------
    if cfg.baseline == "mean":
        # With every reward identical this is uniformly zero and the policy term
        # becomes a silent no-op. Reported below rather than left to be discovered.
        adv = rewards - rewards.mean()
    elif cfg.baseline == "none":
        adv = rewards
    else:
        raise ValueError(f"unknown baseline {cfg.baseline!r}")

    if cfg.adv_clip is not None:
        adv = adv.clamp(-cfg.adv_clip, cfg.adv_clip)

    pg = -(adv.detach() * chosen_logprob).mean()

    # --- exact KL anchor ---------------------------------------------------
    kl = torch.zeros((), dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    if ref_logprobs is not None:
        kl = exact_kl(policy_logprobs, ref_logprobs)

    loss = pg + cfg.kl_beta * kl

    if cfg.entropy_beta:
        pi = policy_logprobs.exp()
        entropy = -(pi * policy_logprobs).sum(dim=-1)
        loss = loss - cfg.entropy_beta * entropy.mean()

    diag = {
        "pg_term": float(pg),
        "kl_term": float(kl),
        "mean_reward": float(rewards.mean()) if rewards.numel() else 0.0,
        "reward_std": float(rewards.std()) if rewards.numel() > 1 else 0.0,
        "adv_abs_mean": float(adv.abs().mean()) if adv.numel() else 0.0,
        "degenerate_rewards": bool(rewards.numel() == 0 or torch.allclose(rewards, rewards[0])),
    }
    return loss, diag
