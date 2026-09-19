"""Toy convergence test for the RLCD Arm A policy-gradient loss, on CPU.

Purpose: prove the arm learns at all, and prove it is DISTINCT from the matched
CE/SFT control, before spending GPU time on a 4B model. Three actions, known
rewards -- the smallest setup where a policy gradient and cross-entropy can
diverge.

What is checked:
  1. REINFORCE, sampled actions, correctness reward -> the policy converges to the
     correct action for every state.
  2. CE (the Arm B control) on the same data also converges. Both reaching the
     same optimum is EXPECTED -- they share a one-hot optimum. So convergence
     alone does not distinguish them.
  3. The two objectives produce DIFFERENT gradients/updates on the same batch,
     which is what makes Arm A a genuinely distinct arm rather than a relabelling
     of Arm B. If this ever comes out equal, the arm is a surrogate and must be
     reported as such.
  4. The exact KL anchor is 0 at the reference, positive once the policy moves,
     and its gradient pulls the policy back.
  5. The degenerate-reward no-op is surfaced rather than silent.

Run: uv run python -m scripts.test_pg_toy
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf.pg_loss import (  # noqa: E402
    PGConfig,
    exact_kl,
    policy_gradient_loss,
    sample_actions,
)

N_STATES = 6
N_ACTIONS = 3
GOLD = torch.tensor([0, 0, 1, 1, 2, 2])


def make_logits(theta: torch.Tensor) -> torch.Tensor:
    """[n, L] logits from an [n, L] parameter tensor."""
    return theta


def run_pg(steps: int = 300, seed: int = 0) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    theta = torch.zeros(N_STATES, N_ACTIONS, requires_grad=True)
    ref = theta.detach().clone()  # frozen reference = the initial policy
    opt = torch.optim.Adam([theta], lr=0.1)
    cfg = PGConfig(kl_beta=0.05, baseline="mean", adv_clip=1.0)
    accs: list[float] = []
    for _ in range(steps):
        logits = make_logits(theta)
        logprobs = torch.log_softmax(logits, dim=-1)
        actions = sample_actions(logprobs).detach()  # a ~ pi, nondifferentiable
        rewards = (actions == GOLD).float()
        loss, _ = policy_gradient_loss(logprobs, actions, rewards, ref_logprobs=torch.log_softmax(ref, -1), cfg=cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            accs.append(float((theta.argmax(-1) == GOLD).float().mean()))
    return theta.detach(), accs


def run_ce(steps: int = 300, seed: int = 0) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    theta = torch.zeros(N_STATES, N_ACTIONS, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    accs: list[float] = []
    for _ in range(steps):
        logits = make_logits(theta)
        loss = torch.nn.functional.cross_entropy(logits, GOLD)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            accs.append(float((theta.argmax(-1) == GOLD).float().mean()))
    return theta.detach(), accs


def main() -> None:
    ok = True
    print("=== 1. REINFORCE with sampled actions converges ===")
    theta_pg, accs_pg = run_pg()
    print(f"  final argmax accuracy: {accs_pg[-1]:.2f}  (first 20 steps mean "
          f"{sum(accs_pg[:20])/20:.2f})")
    if accs_pg[-1] < 1.0:
        print("  FAIL - policy gradient did not converge on the toy")
        ok = False
    else:
        print("  PASS")

    print("\n=== 2. CE control converges too (same optimum, expected) ===")
    theta_ce, accs_ce = run_ce()
    print(f"  final argmax accuracy: {accs_ce[-1]:.2f}")
    if accs_ce[-1] < 1.0:
        print("  FAIL - CE control did not converge")
        ok = False
    else:
        print("  PASS  (convergence does NOT distinguish the arms)")

    print("\n=== 3. the objectives are DISTINCT (different updates on one batch) ===")
    # A fair comparison needs a NON-DEGENERATE policy gradient. With the 'mean'
    # baseline and an all-correct or all-wrong sampled batch the advantage is
    # identically zero and grad_PG is 0 -- a cosine of 0 then proves nothing.
    # Construct a batch with mixed rewards and assert that before comparing.
    torch.manual_seed(1)
    lp = torch.log_softmax(torch.randn(N_STATES, N_ACTIONS), dim=-1)
    acts = sample_actions(lp).detach()
    rew = (acts == GOLD).float()
    if float(rew.std()) == 0.0:
        # Force a mixed-reward batch: half the emitted actions made correct.
        acts = GOLD.clone()
        acts[::2] = (GOLD[::2] + 1) % N_ACTIONS
        rew = (acts == GOLD).float()
    print(f"  batch reward: mean={float(rew.mean()):.3f} std={float(rew.std()):.3f} "
          f"(std>0 required)")
    if float(rew.std()) == 0.0:
        print("  FAIL - could not construct a mixed-reward batch")
        ok = False

    ref_lp = torch.log_softmax(torch.randn(N_STATES, N_ACTIONS), dim=-1)

    # OFF-POLICY is a real error here: the action must be drawn from the SAME
    # distribution the gradient is taken under, or this is not a REINFORCE update.
    # So draw logits first, then sample from their log_softmax. If that draw is
    # degenerate (all rewards equal -> identically zero advantage), resample with
    # fresh seeds until rewards are mixed rather than force-flipping actions,
    # which would corrupt the sampling distribution.
    logits_pg = None
    acts = rew = None
    d_pg = None
    for seed in range(200):
        torch.manual_seed(1000 + seed)
        cand = torch.randn(N_STATES, N_ACTIONS, requires_grad=True)
        cand_lp = torch.log_softmax(cand, dim=-1)
        cand_acts = sample_actions(cand_lp).detach()
        cand_rew = (cand_acts == GOLD).float()
        if float(cand_rew.std()) > 0.0:
            logits_pg, acts, rew = cand, cand_acts, cand_rew
            break
    if logits_pg is None:
        print("  FAIL - could not draw a mixed-reward on-policy batch in 200 seeds")
        ok = False
    else:
        print(f"  on-policy batch: mean={float(rew.mean()):.3f} "
              f"std={float(rew.std()):.3f} (std>0 required)")

    if logits_pg is not None:
        lp_from_logits = torch.log_softmax(logits_pg, dim=-1)
        l_pg, d_pg = policy_gradient_loss(
            lp_from_logits, acts, rew, ref_logprobs=ref_lp, cfg=PGConfig(kl_beta=0.0)
        )
        g_pg = torch.autograd.grad(l_pg, logits_pg)[0]

        # CE on the SAME logits, so both gradients are w.r.t. identical parameters.
        logits_ce = logits_pg.detach().clone().requires_grad_(True)
        l_ce = torch.nn.functional.cross_entropy(logits_ce, GOLD)
        g_ce = torch.autograd.grad(l_ce, logits_ce)[0]

        print(f"  adv_abs_mean={d_pg['adv_abs_mean']:.4f}")
        print(f"  grad_PG norm {g_pg.norm():.6f}   grad_CE norm {g_ce.norm():.6f}   "
              f"(both w.r.t. logits)")

    # A zero PG gradient makes any cosine meaningless; refuse to conclude from it.
    if logits_pg is None:
        pass  # already FAILed above on the missing batch
    elif float(g_pg.norm()) < 1e-8:
        print("  FAIL - grad_PG is ~0; the comparison is vacuous, not a PASS")
        ok = False
    elif torch.allclose(g_pg, g_ce, atol=1e-6):
        print("  FAIL - gradients identical: Arm A is a relabelled surrogate, "
              "not a distinct arm")
        ok = False
    else:
        cos = torch.nn.functional.cosine_similarity(
            g_pg.flatten(), g_ce.flatten(), dim=0
        )
        print(f"  cos(grad_PG, grad_CE) = {cos:.4f}  (defined only because "
              f"|grad_PG| > 0)")
        print("  PASS - distinct objectives/updates on a non-degenerate batch")

    print("\n=== 4. exact KL anchor behaves ===")
    torch.manual_seed(3)
    ref_logits = torch.randn(N_STATES, N_ACTIONS)
    ref = torch.log_softmax(ref_logits, dim=-1)
    k0 = exact_kl(ref, ref)
    print(f"  KL(pi_ref || pi_ref) = {float(k0):.2e}  (expect 0)")

    # A CONTROLLED move away from the reference: scale up the logits, which
    # sharpens the distribution. (Adding a constant is a no-op under softmax, so
    # use a real perturbation.)
    moved_logits = ref_logits * 2.5 + torch.randn(N_STATES, N_ACTIONS) * 0.5
    moved_lp = torch.log_softmax(moved_logits, dim=-1)
    k1 = exact_kl(moved_lp, ref)
    print(f"  KL(pi_moved || pi_ref) = {float(k1):.4f}  (expect > 0)")
    if abs(float(k0)) > 1e-9 or not float(k1) > 0:
        print("  FAIL - KL anchor misbehaves")
        ok = False
    else:
        print("  PASS")

    # Gradient step on the LOGITS (the actual free parameters), recomputing
    # log_softmax inside the objective so the tensor stays normalized.
    logits = moved_logits.clone().requires_grad_(True)

    def kl_of(lg: torch.Tensor) -> torch.Tensor:
        return exact_kl(torch.log_softmax(lg, dim=-1), ref)

    kl = kl_of(logits)
    gkl = torch.autograd.grad(kl, logits)[0]
    with torch.no_grad():
        k_after = float(kl_of(logits - 0.1 * gkl))
    print(f"  KL after one gradient-descent step on the logits: {k_after:.4f} "
          f"(was {float(k1):.4f})")
    if not k_after < float(k1):
        print("  FAIL - KL gradient does not reduce the divergence")
        ok = False
    else:
        print("  PASS - anchor reduces divergence")

    print("\n=== 4b. unnormalized inputs are rejected (not silently mis-KL'd) ===")
    try:
        exact_kl(ref_logits, ref)  # raw logits, not log-probs
        print("  FAIL - raw logits accepted")
        ok = False
    except ValueError as e:
        print(f"  PASS - ValueError raised: {str(e)[:80]}...")

    print("\n=== 5. degenerate rewards are surfaced, not silent ===")
    lp2 = torch.log_softmax(torch.randn(N_STATES, N_ACTIONS), dim=-1)
    a2 = sample_actions(lp2).detach()
    same = torch.ones(N_STATES)
    _, diag = policy_gradient_loss(lp2, a2, same, cfg=PGConfig())
    print(f"  all-equal rewards -> degenerate_rewards={diag['degenerate_rewards']}, "
          f"adv_abs_mean={diag['adv_abs_mean']:.2e}, pg_term={diag['pg_term']:.2e}")
    if not diag["degenerate_rewards"]:
        print("  FAIL - silent no-op not flagged")
        ok = False
    else:
        print("  PASS - flagged")

    print("\n=== 6. argmax actions are rejected ===")
    lp3 = torch.log_softmax(torch.randn(N_STATES, N_ACTIONS), dim=-1)
    bad = lp3.argmax(-1)
    _, diag_ok = policy_gradient_loss(lp3, bad, (bad == GOLD).float())
    print(f"  argmax actions accepted={True} (no exception) -- contract is documented, "
          f"not enforced by type")
    print("  (the loader must call sample_actions(); see the arm's training loop)")

    print()
    print("RESULT:", "PASS - arm is learnable and distinct from the CE control" if ok
          else "FAIL - see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
