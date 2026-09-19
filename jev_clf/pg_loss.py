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
    # 'batch_mean' | 'none'. 'batch_mean' subtracts the batch mean reward (the
    # cheap variance reduction). The old name 'mean' was ambiguous and 'batch'
    # silently aliased 'none', which would run the wrong estimator; both are now
    # rejected explicitly. A learned value head is out of scope here.
    baseline: str = "batch_mean"
    # Advantage clipping, so one lucky batch cannot take a huge step.
    adv_clip: float | None = 1.0
    # Entropy bonus. A positive value FLATTENS the policy (loss -= beta*H), which
    # DISCOURAGES collapse and over-confidence -- the opposite of the earlier note
    # here, which had it backwards. Kept OFF by default for two other reasons: on
    # a 3-way label space it can cost accuracy, and deliberately flattening the
    # distribution distorts the calibrated probabilities we are trying to report.
    # Enabling it is a deliberate trade, not a free win.
    entropy_beta: float = 0.0


def exact_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    atol: float = 1e-2,
    check_normalized: bool | None = None,
) -> torch.Tensor:
    """Exact KL(pi || pi_ref) over the label space. Both inputs are [n, L] LOG-probs.

    Computed over the full support, so it is correct regardless of which action the
    policy emitted. Returns a scalar mean over the batch.

    Both inputs must be VALID normalized log-distributions (exp sums to 1 per row):
    passing raw logits yields a number that is not a KL divergence at all.

    The check is DEBUG-ONLY. It costs a `torch.allclose` and therefore a device
    synchronisation every call, which is unacceptable in the training hot path. It
    defaults to on when grad is disabled (tests, eval) and off during training.
    The tolerance is loose by default because bf16/fp16 log_softmax rows do not
    sum to 1 within 1e-4; sums are upcast to float32 before comparing.
    """
    if policy_logprobs.shape != ref_logprobs.shape:
        raise ValueError("policy and reference log-probs must have the same shape")
    if policy_logprobs.ndim != 2:
        raise ValueError("expected [n, L] log-prob tensors")
    if check_normalized is None:
        check_normalized = not torch.is_grad_enabled()
    if check_normalized:
        for name, t in (("policy_logprobs", policy_logprobs), ("ref_logprobs", ref_logprobs)):
            sums = t.exp().sum(dim=-1).float()  # upcast: bf16/fp16 rows drift
            if not torch.allclose(sums, torch.ones_like(sums), atol=atol):
                worst = float((sums - 1.0).abs().max())
                raise ValueError(
                    f"{name} rows do not sum to 1 (max deviation {worst:.4g}); "
                    "pass log_softmax(logits), not raw logits or drifted tensors"
                )
    pi = policy_logprobs.exp()
    return (pi * (policy_logprobs - ref_logprobs)).sum(dim=-1).mean()


def exact_reward_loss(
    policy_logprobs: torch.Tensor,
    gold: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
    cfg: PGConfig | None = None,
    soft_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact expected-reward objective: the analytically marginalized bandit loss.

        hard gold :  L = -mean_s( pi(gold | s) )        + beta * KL
        soft target: L = -mean_s( sum_a q(a|s) pi(a|s) ) + beta * KL

    REWARD SOURCE — the thing to get right. The reward must mean *ground truth
    correctness*, so:

      * human-labelled rows -> `gold` (the hard human label). Unambiguous.
      * teacher-labelled rows -> `soft_target` = the teacher's FULL distribution
        q(a|s), giving E_q[R] = sum_a q(a) pi(a). Using the teacher's argmax as if
        it were ground truth would optimise imitation of a teacher that is itself
        ~0.83 accurate on this data, and would throw away the uncertainty the
        teacher actually expressed (e.g. {refuted 0.93, NEI 0.07}). The soft form
        keeps that signal and is the only defensible way to use teacher rows here.

    Passing neither `gold` nor `soft_target` is an error: there is no reward.


    """
    cfg = cfg or PGConfig()
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    n, L = policy_logprobs.shape
    if (gold is None) == (soft_target is None):
        raise ValueError("pass exactly one of gold (hard) or soft_target (distribution)")

    pi = policy_logprobs.exp()

    if gold is not None:
        if gold.shape != (n,):
            raise ValueError(f"gold must be [{n}]")
        if gold.numel() and (int(gold.min()) < 0 or int(gold.max()) >= L):
            raise ValueError("gold index out of range for the label space")
        expected_r = pi.gather(1, gold.view(-1, 1)).squeeze(1)
        source = "hard_gold"
    else:
        assert soft_target is not None
        if soft_target.shape != (n, L):
            raise ValueError(f"soft_target must be [{n}, {L}]")
        # q need not be normalized to be a valid expectation, but a non-normalized
        # target is almost certainly a bug (logits passed instead of probabilities).
        row_sums = soft_target.sum(dim=-1).float()
        if soft_target.numel() and not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2
        ):
            raise ValueError(
                "soft_target rows must sum to 1; pass probabilities, not logits"
            )
        expected_r = (soft_target * pi).sum(dim=-1)
        source = "soft_teacher"

    reward_term = -expected_r.mean()

    kl = torch.zeros((), dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    if ref_logprobs is not None:
        kl = exact_kl(policy_logprobs, ref_logprobs)

    loss = reward_term + cfg.kl_beta * kl

    if cfg.entropy_beta:
        entropy = -(pi * policy_logprobs).sum(dim=-1)
        loss = loss - cfg.entropy_beta * entropy.mean()

    diag = {
        "reward_source": source,  # type: ignore[dict-item]
        "reward_term": float(reward_term.detach()),
        "kl_term": float(kl.detach()),
        "mean_expected_reward": float(expected_r.detach().mean()),
        "min_expected_reward": float(expected_r.detach().min()) if expected_r.numel() else 0.0,
        "saturated_frac": float((expected_r.detach() > 0.99).float().mean())
        if expected_r.numel() else 0.0,
    }
    return loss, diag

    """
    cfg = cfg or PGConfig()
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    n, L = policy_logprobs.shape
    if (gold is None) == (soft_target is None):
        raise ValueError("pass exactly one of gold (hard) or soft_target (distribution)")

    pi = policy_logprobs.exp()

    if gold is not None:
        if gold.shape != (n,):
            raise ValueError(f"gold must be [{n}]")
        if gold.numel() and (int(gold.min()) < 0 or int(gold.max()) >= L):
            raise ValueError("gold index out of range for the label space")
        expected_r = pi.gather(1, gold.view(-1, 1)).squeeze(1)
        source = "hard_gold"
    else:
        assert soft_target is not None
        if soft_target.shape != (n, L):
            raise ValueError(f"soft_target must be [{n}, {L}]")
        # q need not be normalized to be a valid expectation, but a non-normalized
        # target is almost certainly a bug (logits passed instead of probabilities).
        row_sums = soft_target.sum(dim=-1).float()
        if soft_target.numel() and not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2
        ):
            raise ValueError(
                "soft_target rows must sum to 1; pass probabilities, not logits"
            )
        expected_r = (soft_target * pi).sum(dim=-1)
        source = "soft_teacher"

    reward_term = -expected_r.mean()

    kl = torch.zeros((), dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    if ref_logprobs is not None:
        kl = exact_kl(policy_logprobs, ref_logprobs)

    loss = reward_term + cfg.kl_beta * kl

    if cfg.entropy_beta:
        entropy = -(pi * policy_logprobs).sum(dim=-1)
        loss = loss - cfg.entropy_beta * entropy.mean()

    diag = {
        "reward_source": source,  # type: ignore[dict-item]
        "reward_term": float(reward_term.detach()),
        "kl_term": float(kl.detach()),
        "mean_expected_reward": float(expected_r.detach().mean()),
        "min_expected_reward": float(expected_r.detach().min()) if expected_r.numel() else 0.0,
        "saturated_frac": float((expected_r.detach() > 0.99).float().mean())
        if expected_r.numel() else 0.0,
    }
    return loss, diag

    Relationship to CE, stated precisely because a sloppy version is tempting.
    With z_gold the correct-class logit:

      CE             objective  log pi(gold);  d/dz_gold = p_gold - 1
      exact reward   objective  pi(gold);      d/dz_gold = -p_gold * (1 - p_gold)

    Both objectives drive p_gold toward 1 and both are zero-gain once p_gold = 1,
    so "CE saturates and reward does not" is WRONG. The difference that matters is
    at CONFIDENTLY WRONG examples:

      CE:            |grad| = 1 - p_gold          -> approaches 1 as p_gold -> 0
      exact reward:  |grad| = p_gold*(1 - p_gold) -> approaches 0 as p_gold -> 0
                                                     (peak only at p_gold = 0.5)

    So this arm's gradient VANISHES exactly on the hard, confidently-wrong
    examples -- and the not_enough_info rows we are trying to fix are precisely
    those. Zero variance is NOT the same as better optimisation: this objective is
    lower-variance but has a pathological flat region where the data most needs a
    signal. Measure it (scripts/test_pg_toy.py section 7), do not assume it away.

    On the KL anchor as a mitigation: it does NOT rescue this. The anchor pulls
    toward the FROZEN REFERENCE, and if the reference already assigns p_gold ~ 0
    for a hard example -- which is exactly the case for the rows this arm is meant
    to fix -- then the KL term actively holds the policy there. The anchor bounds
    drift; it does not restore gradient on the flat region. Whether it helps or
    hurts must be measured against the cached reference probabilities.
    """
    cfg = cfg or PGConfig()
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    n, L = policy_logprobs.shape
    if gold.shape != (n,):
        raise ValueError(f"gold must be [{n}]")
    if gold.numel() and (int(gold.min()) < 0 or int(gold.max()) >= L):
        raise ValueError("gold index out of range for the label space")

    pi = policy_logprobs.exp()
    p_gold = pi.gather(1, gold.view(-1, 1)).squeeze(1)
    reward_term = -p_gold.mean()

    kl = torch.zeros((), dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    if ref_logprobs is not None:
        kl = exact_kl(policy_logprobs, ref_logprobs)

    loss = reward_term + cfg.kl_beta * kl

    if cfg.entropy_beta:
        entropy = -(pi * policy_logprobs).sum(dim=-1)
        loss = loss - cfg.entropy_beta * entropy.mean()

    diag = {
        "reward_term": float(reward_term.detach()),
        "kl_term": float(kl.detach()),
        "mean_p_gold": float(p_gold.mean()),
        "min_p_gold": float(p_gold.min()) if p_gold.numel() else 0.0,
        "saturated_frac": float((p_gold > 0.99).float().mean()) if p_gold.numel() else 0.0,
    }
    return loss, diag


def bandit_train_step(
    policy_logprobs: torch.Tensor,
    gold: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
    cfg: PGConfig | None = None,
    mode: str = "exact_reward",
    soft_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The production training step. Owns sampling; the caller cannot inject actions.

    This exists because "the caller must remember to sample" is not an enforceable
    contract: an integer index tensor carries no grad whether it came from argmax
    or from `Categorical.sample()`, so no tensor-level check can tell them apart.
    The only real enforcement is to not expose an actions argument at all.

    `mode`:
      'exact_reward' (default, primary) -- L = -mean pi(gold) + beta*KL. The reward
          is enumerable over the label space, so this is the exact expected-reward
          objective: zero sampling variance, no baseline, no degenerate batches.
          Caveat: its gradient vanishes on confidently-wrong rows; measure it.
      'reinforce' -- sampled-action policy gradient, kept for parity with rewards
          that cannot be enumerated. Actions are drawn HERE, inside the step.

    Returns (loss, diagnostics) with scalar diagnostics detached, so no
    requires-grad warning fires and no graph is retained by reporting.
    """
    cfg = cfg or PGConfig()
    if mode == "exact_reward":
        loss, diag = exact_reward_loss(
            policy_logprobs, gold=gold, ref_logprobs=ref_logprobs, cfg=cfg,
            soft_target=soft_target,
        )
    elif mode == "reinforce":
        if gold is None:
            raise ValueError(
                "reinforce mode needs a hard `gold`; a soft target cannot supply a "
                "sampled-action reward"
            )
        # Sampling happens inside the step, which is the point of this helper.
        actions = sample_actions(policy_logprobs).detach()
        rewards = (actions == gold).float()
        loss, diag = policy_gradient_loss(
            policy_logprobs, actions, rewards, ref_logprobs, cfg
        )
        diag = dict(diag)
        diag["sampled_action_rate"] = float(rewards.mean())
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return loss, {k: (float(v) if isinstance(v, (int, float)) or torch.is_tensor(v) else v)
                  for k, v in diag.items()}


def sample_actions(
    policy_logprobs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw actions a ~ Categorical(pi(.|s)). REQUIRED by the arm.

    REINFORCE's estimator is unbiased only for actions sampled from the policy.
    argmax gives a different (biased) quantity and would make the arm a mislabeled
    surrogate. Note no tensor-level check can detect argmax, because an integer
    index carries no grad either way -- which is why bandit_train_step() samples
    internally instead of accepting actions from the caller.

    Implemented with torch.multinomial rather than Categorical.sample(), because
    Categorical.sample() accepts no generator; this way a seeded generator makes
    runs genuinely reproducible.
    """
    if policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must be [n, L]")
    probs = policy_logprobs.exp()
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


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
    if cfg.baseline == "batch_mean":
        # With every reward identical this is uniformly zero and the policy term
        # becomes a silent no-op. Reported below rather than left to be discovered.
        adv = rewards - rewards.mean()
    elif cfg.baseline == "none":
        adv = rewards
    else:
        raise ValueError(
            f"unknown baseline {cfg.baseline!r}; use 'batch_mean' or 'none'"
        )

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
        "pg_term": float(pg.detach()),
        "kl_term": float(kl.detach()),
        "mean_reward": float(rewards.mean()) if rewards.numel() else 0.0,
        "reward_std": float(rewards.std()) if rewards.numel() > 1 else 0.0,
        "adv_abs_mean": float(adv.abs().mean()) if adv.numel() else 0.0,
        "degenerate_rewards": bool(rewards.numel() == 0 or torch.allclose(rewards, rewards[0])),
    }
    return loss, diag
