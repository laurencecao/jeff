# RLCD arm — plan (open approximation)

## What RLCD actually is, per the only primary source

TypeSafe's own site states Jev uses "a new architecture, a new sampler, and a new
training algorithm: **Reinforcement Learning for Calibrated Decisions (RLCD)**"
(https://typesafe.ai/, fetched 2026-09-19). It is positioned explicitly *against*
RLHF, which they blame for "mode dropping, overconfidence, and lack of
reliability". **They do not disclose the mechanics.**

Two corrections this fixes:

1. RLCD here is **not** "Reinforcement Learning from Contrastive Distillation"
   (Yang et al.). That is a different method and the name collision is a trap.
2. Because no spec is published, anything built here is **an open approximation
   targeting Jev's observable contract** — typed decisions plus calibrated
   confidence — not a reimplementation. Nothing in this arm should be described as
   "Jev's method".

## The hard obstacle, measured

A calibrated-decision objective needs outcomes that are genuinely uncertain
*conditional on the input the model sees*. For a classifier whose input is
`(claim, evidence)` and whose output is a label, that means repeated annotation of
the **same** `(claim, evidence)`.

Measured on the corpus we have, keyed by the full task
`(state, question_id, schema_id, label_source)`:

| asset | distinct full-task keys | keys with repeats | keys with disagreeing labels |
|---|---|---|---|
| `ground_truth.jsonl` train | 1,589 | 0 | 0 |
| `sft_train_multi.jsonl` | 12,119 | 0 | 0 |

**There is no repeated-label data.** An earlier count of "848 disagreeing states"
was an artifact: it collapsed across question kinds, and `sft_train_multi`
deliberately asks Choice, Noul and Score questions over the same state, whose
labels are not comparable. Keyed correctly, the disagreement vanishes.

Consequences, stated plainly:

- A per-sample reward of `1[correct]` is **ordinary supervised cross-entropy**
  under a policy-gradient wrapper. It carries no calibration information and
  actively pushes toward confidence 1.
- Using the model's **own sampled** correctness fraction as a calibration target
  is circular — it measures the policy's own error rate, not the truth.
- `log p(y_gold)` / Brier on a deterministic label is likewise CE, not a proper
  score over uncertain outcomes.

So a faithful "RL for calibrated decisions" reward is **not constructible from
this corpus**. Building one anyway would be the exact overconfidence failure the
name is meant to prevent.

## What is therefore in scope

Split the two things that RLCD fuses, and be explicit about the seam:

**Arm A (primary, RL).** Optimize **decision correctness** with an actual policy
objective on train-only states. Reward = correctness of the emitted label.
Objective = policy gradient with a frozen reference for a KL anchor (GRPO-style
group-relative advantage, or REINFORCE with a baseline). This is genuine RL, and
it targets the measured NEI-recall deficit.

**Calibration is then handled separately, not smuggled into the reward:** post-hoc
calibration (temperature scaling / isotonic) fitted on a **dedicated calibration
carve taken from the legal train states**, and *evaluated* on untouched states.

The carve is now a real artifact, not prose:
`scripts/make_calib_carve.py` writes `data/factcheck/calib_carve.jsonl` (238 rows)
and `calib_carve_ids.json` (the row and group ids), so downstream code can
**exclude** these rows mechanically.

Verified properties: drawn only from `split='train'`; **238 rows across 198
groups**, group-disjoint from the remaining 1,007 groups (asserted, not assumed);
class mix 104 supported / 80 NEI / 54 refuted. Those 238 rows are excluded from
both Arm A and Arm B.

Note the tooling caveat: `train.py` already reports `val_ece_temperature` and
`val_ece_isotonic`, but it fits them **on the sealed val split**. That is fine for
the legacy SFT loop, where val is only a smoke test, but it is **not** acceptable
here: fitting on val and then reporting val ECE is a leak. This arm fits on the
train-derived carve and reports val/test/scale untouched.

**Arm B (control, mandatory).** Matched **chosen-only SFT** on the same rows, same
optimizer budget, same effective batch. Without this, any gain is attributable to
"more hard negatives" or "more steps" rather than to the policy objective.

**Abstention.** An explicit `not_enough_info` action, since NEI recall is the
deficit. Reward shaping must state whether abstention is rewarded or merely
permitted.

## Data discipline (non-negotiable)

- Training states come **only** from `split='train'` rows.
- `eval_large.jsonl` (9,730) and the 199-row val split are **evaluation-only**.
  Mining the measured errors into training would invalidate the headline A/B and
  every published number.
- The absent / overstated / tied probes stay **untouched gates**.
- A fail-closed disjointness gate runs before any training:
  `scripts/check_train_eval_disjoint.py`. Current status: **PASS** — 0 exact
  `(claim, ordered evidence)` overlaps and 0 `group_id` overlaps against
  `ground_truth` (val+test), `eval_large`, `eval_schemas`, `fixture`. It parses
  the actual SFT prompt states (12,119/12,119 parsed) rather than trusting
  `row_id` naming or `gt_row_id`.
- New training rows, if synthesized, must extend that gate rather than bypass it.

## Where new contrastive material may come from

On each **train-only** state, run the model under near-identical *strict* vs
*permissive* guideline prompts; keep a pair only when the strict variant agrees
with train gold and the permissive one differs. Train preference on the original
unmodified prompt. This preserves a contrastive pair-construction step without
inventing states or touching eval rows. Held-out guideline pairs and groups are
reserved for validation.

Honest limitation: because the model emits a single label token, such pairs can
degenerate into trivial `gold > wrong-label`. If they do, that is **RLCD-style
data generation + preference learning**, and must be labelled as that — not as
RL over a calibrated reward.

## First bounded experiment

1. Build the train-only pool: 12,119 rows, gated disjoint.
2. Generate contrastive/negative material for the three failing shapes
   (absent, tied, overstated evidence) on train-only states.
3. Score with live Jev as teacher/reward on those **new** rows only.
4. Train **Arm A** (policy gradient, correctness reward, KL anchor) from the
   current `lora_4b_multi` adapter.
5. Train **Arm B** (chosen-only SFT control) on the same rows and budget.
6. Evaluate only on untouched `val` (n=199), the 9,730 scale split, and the
   held-out probes. Report accuracy, ECE, and the three probe families.
7. Gate Noul/Score and six-shape capability parity afterwards — choice-only
   preference tuning can regress the multi-primitive behaviour.

## Compute constraint

`trl` is not installed; the local box is an M3 Max with 36 GB unified memory,
where a 4B LoRA at batch 4 x 2048 already OOM-killed at step 2. **This arm needs
GPU.** Colab was returning no sessions as of this writing (A100 reclaimed within
minutes; per-epoch checkpointing and `--resume` are in place so a recycle costs at
most one epoch).

## Failure conditions (stated before running)

- Arm A does not beat Arm B on untouched scale accuracy → the policy objective is
  not the lever; report it and stop.
- Arm A improves accuracy while ECE degrades beyond noise → correctness and
  calibration are trading off, and the separation above was the wrong cut.
- Any capability regression on Noul/Score or the six shapes → the arm is not
  shippable regardless of its accuracy.
