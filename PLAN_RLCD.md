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

## What the corpus does and does not constrain

A calibrated-decision objective needs a target that distinguishes confidence from
accuracy. Two different things were being conflated, and the distinction matters:

**Calibration IS learnable from singly-labelled data.** Log loss
(cross-entropy) is a *strictly proper scoring rule*: its population minimiser
over scoring functions is the true conditional probability `P(y|x)`. Repeated
inputs are **not** required to learn calibrated probabilities — that is the
standard result, and it is how every calibrated classifier is trained in
practice. So there is no impossibility here, and an earlier draft of this file
wrongly claimed there was.

**What the corpus lacks** is outcome *stochasticity*. Keyed by the full task
`(state, question_id, schema_id, label_source)`:

| asset | distinct full-task keys | keys with repeats | keys with disagreeing labels |
|---|---|---|---|
| `ground_truth.jsonl` train | 1,589 | 0 | 0 |
| `sft_train_multi.jsonl` | 12,119 | 0 | 0 |

An earlier count of "848 disagreeing states" was an artifact: it collapsed across
question kinds, and `sft_train_multi` deliberately asks Choice, Noul and Score
questions over the same state, whose labels are not comparable. Keyed correctly,
the disagreement vanishes.

Consequences, precisely:

- Any reward that depends on *sampling an outcome* (e.g. "was the emitted label
  right?") has no signal beyond the label itself here, because every label is
  single and deterministic. That limits which reward shapes are *usable*.
- Instance-level calibration cannot be *empirically verified* without repeats.
- **Population-level calibration remains measurable** on held-out data — that is
  exactly what ECE does, and it is what this project already reports. So
  "calibration is unmeasurable here" would also be false.

## Correctness reward vs cross-entropy

These are not the same objective, and the difference is worth stating because
both are on the table:

- A correctness policy gradient maximises `p_y` for the emitted label.
- Cross-entropy maximises `log p_y`.

They share the **same one-hot optimum** on deterministic-label data, but they are
different objectives with different gradients and different optimisation paths.
So "correctness RL is just CE with extra steps" is too strong. What is true is
that neither, on its own, is a *proper scoring rule over sampled outcomes* here —
because there are no sampled outcomes in this corpus. CE is a proper scoring rule
over the label distribution, which is why it is the right default target for
calibration and why an RL arm must justify itself against it rather than assume
superiority.

## The one bounded experiment (authoritative)

Everything below is the experiment. Contrastive-prompt pair generation is NOT part
of it — that belongs to the *other* RLCD acronym (Yang et al.) and supplies no
correctness reward for TypeSafe-style RLCD. Live Jev is NOT used as the reward:
imitating a teacher that is itself only ~0.83 accurate on this data would train
the model toward the teacher's errors. Gold labels are the reward.

**Source rows.** `data/factcheck/sft_train_multi.jsonl`, `question_id == 'verdict'`,
`split == 'train'`, minus the 198 calibration-carve groups. Measured: **7,826
rows** — **6,021 teacher-labelled** (`label_source='jev-1.13.0'`) plus **1,805
human-labelled** (`label_source='ground_truth'`). Label space is always the 3
verdict labels. (An earlier draft said "7,000 Jev-distilled", which is the count
in the UNCUT pool of 9,119 verdict rows, not the carve-excluded 7,826.)

**Reward source, stated correctly.** Human rows supply a hard `gold`. Teacher rows
supply `soft_target = q`, the teacher's full distribution, giving
`E_q[R] = sum_a q(a) pi(a)`.

But note the limitation, because it is easy to oversell: `-sum_a q(a) pi(a)` is
linear in `pi`, so its optimum is a **one-hot at argmax(q)**. It weights the
gradient by the shape of `q` during training and still collapses to the teacher's
single most likely label at convergence — it does NOT preserve the teacher's
uncertainty. A proper scoring rule that is actually minimised at `pi = q` is soft
cross-entropy `-sum_a q(a) log pi(a)` (or KL(pi || q)), which is what
`scripts/jev_clf_lora_train.py` already implements.

**Objective — Arm A (primary).** The reward is enumerable over 3 labels
(`r(a) = 1[a == gold]`), so the exact expected-reward objective is available and is
the primary arm, not REINFORCE:

    L_A = -mean_s( pi(gold | s) )  +  beta * KL( pi(.|s) || pi_ref(.|s) )

Zero sampling variance, no baseline, no degenerate-batch handling. Its known
weakness must be measured, not assumed away: `d/dz_gold = -p_gold*(1-p_gold)`, so
its gradient vanishes on confidently-wrong rows — exactly the NEI rows this arm
targets. Section 7 of `scripts/test_pg_toy.py` measures the gradient at
`p_gold` in {0.001, 0.5, 0.999}. REINFORCE is kept only as a parity check.

**Objective — Arm B (control, mandatory).** Plain cross-entropy on the same rows,
same optimizer, same effective batch, same epochs. CE's gradient is `p_gold - 1`,
i.e. it does NOT vanish on hard errors, so this is a real competitor rather than a
straw man. Without Arm B no gain can be attributed to the objective.

**Calibration.** Not in the reward. Post-hoc temperature/isotonic fitted on the
238-row carve only, then applied unchanged to val/test/scale. `train.py` currently
fits these on the sealed val split, which is a leak for this arm and must not be
reused; fitting on val and reporting val ECE is circular.

**Metrics, all on untouched data.** Scale accuracy (n=9,730) as primary; scale ECE;
NEI recall on single-passage rows; the three probe families (conjunction,
granularity, jaggedness); and the capability-parity check (Noul/Score + six
shapes) because choice-only preference tuning can regress multi-primitive
behaviour.

**Reported alongside each other, never collapsed:** Arm A vs Arm B vs the current
`lora_4b_multi` baseline, on the same rows, with paired McNemar.

**Abstention.** `not_enough_info` is a normal label in the space, so it is
permitted by construction. Whether it should be *rewarded* beyond correctness is
explicitly out of scope for this first arm — the honest default is to reward
correctness only and see what the NEI recall does.

**Not in this experiment:** contrastive prompt pairs, reward models, PPO, live-Jev
reward, and mining `eval_large` errors into training.

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

1. Emit the two arm datasets from the gated pool (same rows, one artifact each):
   Arm A and Arm B see identical rows; only the objective differs.
2. Cache the frozen reference's full `[n, L]` log-prob vector per row, once, from
   `artifacts/jev_clf/lora_4b_multi`. Never hold policy and reference in memory
   simultaneously — that is what the cache is for.
3. Train **Arm A** (`-mean pi(gold) + beta*KL`) from `lora_4b_multi`.
4. Train **Arm B** (matched CE) on the same rows, same optimizer/epochs/batch.
5. Fit temperature/isotonic on the 238-row carve ONLY; apply unchanged downstream.
6. Evaluate on untouched `val` (n=199), the 9,730 scale split, and the held-out
   probes. Report accuracy, ECE, NEI recall, and the three probe families.
7. Gate Noul/Score and six-shape capability parity — choice-only tuning can
   regress multi-primitive behaviour.
8. Report Arm A vs Arm B vs the `lora_4b_multi` baseline with paired McNemar.

## Compute constraint

`trl` is not installed and is not needed — the objective is a pure-tensor loss
plus a custom loop, which is also why it could be verified on CPU first. The local
box is an M3 Max with 36 GB unified memory, where a 4B LoRA at batch 4 x 2048
OOM-killed at step 2, so **the 4B runs need a GPU**. Colab was returning no
sessions as of this writing. Per-epoch checkpointing and `--resume` are in place
so a recycle costs at most one epoch.

## Failure conditions (stated before running)

- Arm A does not beat Arm B on untouched scale accuracy → the policy objective is
  not the lever; report it and stop.
- Arm A improves accuracy while ECE degrades beyond noise → correctness and
  calibration are trading off, and the separation above was the wrong cut.
- Any capability regression on Noul/Score or the six shapes → the arm is not
  shippable regardless of its accuracy.
