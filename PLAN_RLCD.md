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

> **STATUS: the reward arm is DEMOTED, not primary. Do not run it first.**
> Two independent measurements killed it as the lead option:
>
> 1. Its gradient is ~1000x weaker than CE on the confidently-wrong rows this
>    deficit lives in (`|grad|` ratio 0.0010 at `p_gold=0.001`).
> 2. Its soft variant collapses to `argmax(q)` at convergence (measured KL 0.415
>    vs 0.0002 for soft CE on the same target), so it does not preserve teacher
>    uncertainty either.
>
> The better target for the `not_enough_info` deficit is the already-written,
> pre-registered soft-target distillation arm
> (`scripts/jev_clf_lora_train.py`, `PREREGISTRATION_soft_distill.md`), which
> computes `KL(q || pi)` — a proper scoring rule minimised at `pi = q`.
> Everything below is kept as the record of the RL arm's design and its failure.

Contrastive-prompt pair generation is NOT part of this experiment — that belongs
to the *other* RLCD acronym (Yang et al.) and supplies no correctness reward for
TypeSafe-style RLCD.

**Reward source is mixed, not "gold".** Human-labelled rows (1,805) supply a hard
`gold`. Teacher-labelled rows (6,021) supply `soft_target = q`, the teacher's full
distribution, giving `E_q[R] = sum_a q(a) pi(a)`. Live Jev is NOT re-queried as a
reward: the teacher labels already in the data are reused, and a teacher that is
itself ~0.83 accurate is an imperfect correctness signal either way.

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
`scripts/jev_clf_lora_train.py` already implements `KL(q || pi)`.

**Objective — Arm A (DEMOTED; kept for the record).** The reward is enumerable
over 3 labels (`r(a) = 1[a == gold]`), so the exact expected-reward objective is
available in closed form rather than needing REINFORCE:

    L_A = -mean_s( pi(gold | s) )  +  beta * KL( pi(.|s) || pi_ref(.|s) )

Zero sampling variance, no baseline, no degenerate-batch handling. Its known
weakness must be measured, not assumed away: `d/dz_gold = -p_gold*(1-p_gold)`, so
its gradient vanishes on confidently-wrong rows — exactly the NEI rows this arm
targets. Section 7 of `scripts/test_pg_toy.py` measures the gradient at
`p_gold` in {0.001, 0.5, 0.999}. REINFORCE is kept only as a parity check.

**Arm B (control).** Plain cross-entropy on the same rows, same optimizer, same
effective batch, same epochs. CE's gradient is `p_gold - 1`, i.e. it does NOT
vanish on hard errors, so it is a real competitor rather than a straw man. For the
soft-distillation experiment the matched control is **`lora_4b_multi` itself** (see
the matched-A/B decision below) — no separate run is required.

**Calibration — NOT part of the chosen experiment.** The carve-fitted
temperature/isotonic plan below applies to a carve-clean experiment, which is the
REJECTED option: `lora_4b_multi` saw the 238 carve rows, so the carve cannot
calibrate this A/B. The chosen experiment reports **raw ECE**, exactly as the
published 0.0807 already is. Separately, note that `train.py` fits its
temperature/isotonic on the sealed val split — that is fine for the legacy loop
where val is a smoke test, but it must never be quoted as a calibrated val ECE for
this arm, since fitting on val and reporting val is circular.

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

## The experiment to run: soft-distillation A/B (NOT the reward arm)

The reward arm is demoted and is not part of this experiment.

**The one change under test.** `scripts/jev_clf_lora_train.py` already computes
`CE + soft_target_weight * mean_rows KL(q || pi)`. The RL arm is dropped because
two measurements disqualified it: the expected-reward gradient is ~1000x weaker
than CE exactly on confidently-wrong rows (ratio 0.0010 at `p_gold=0.001`), and its
soft form collapses to `argmax(q)` (measured KL 0.415 vs 0.0002 for `KL(q || pi)`
on the same target `q=[0.93,0.07,0.0]`).

### Matched-A/B decision: run on ALL 12,119 rows, no carve

Two ways to do this were on the table and they are NOT compatible. Resolved in
favour of the pre-registered one:

- **CHOSEN — all 12,119 rows, no carve-fitted calibration.** `lora_4b_multi` was
  trained on all 12,119 rows. A soft arm trained on the 10,376 carve-excluded rows
  would differ in BOTH the objective and the training data, so a win could not be
  attributed to the objective, and calling it a matched A/B would be false. It
  also costs one GPU run instead of two. The pre-registration
  (`PREREGISTRATION_soft_distill.md`) was written against exactly this comparison.
- **REJECTED — a carve-clean calibration experiment.** Training both a CE and a
  soft arm on the same 10,376 rows would permit fitting temperature/isotonic on
  the carve without touching either arm's rows. It is the cleaner *calibration*
  experiment but it is a different, larger experiment (two runs, and a new
  baseline that is no longer the published `lora_4b_multi`).

**Consequence, stated plainly:** because `lora_4b_multi` saw the 238 carve rows,
the carve CANNOT be used to calibrate this A/B. So this experiment does **no**
post-hoc calibration at all — it reports raw ECE, exactly as the published 0.0807
already is. That is a real limitation of the pre-registered design, not an
oversight: the carve exists for the RLCD arm, which trains on carve-excluded rows,
and it will be used there if that arm is ever revived.

### Steps

1. Training pool: `data/factcheck/sft_train_multi.jsonl`, `split='train'`, all
   12,119 rows. Leave `data.exclude_calib_carve` at its default `false` so the row
   set is identical to `lora_4b_multi`'s.
2. Train from the base model exactly as `lora_4b_multi` was, with ONE variable
   changed: `optim.soft_target_weight: 1.0`. Same LoRA shape, optimizer, epochs,
   batch, seed. Configs `configs/jev_clf_lora_soft.yaml` (A100) and
   `..._soft_t4.yaml` (T4; batch 1 x accum 32 keeps the effective batch at 32).
3. Matched baseline = **`lora_4b_multi` itself**: byte-identical config except
   `soft_target_weight`, trained on the identical 12,119 rows. No separate control
   run is needed.
4. Expect `soft targets usable on 10319/12119` — the 1,800 score rows share first
   token 220 and legitimately switch the KL term off. A materially lower count
   means the KL path is misconfigured; stop.
5. Score on the untouched 9,730-row scale split and the 199-row val split; run the
   three probe families.
6. Gate Noul/Score and six-shape capability parity — choice-path tuning can regress
   multi-primitive behaviour.
7. Report `lora_4b_soft` vs `lora_4b_multi` with a paired McNemar on the same
   rows, and check the pre-registered prediction (+10..+40 rows, NEI-concentrated,
   probably not individually significant at p<0.05).

## Compute constraint

The local box is an M3 Max with 36 GB unified memory, where a 4B LoRA at batch
4 x 2048 OOM-killed at step 2, so **the run needs a GPU**. Colab was returning no
sessions as of this writing. Per-epoch checkpointing and `--resume` are in place,
and `scripts/colab_run_soft.sh` packs the adapter after every epoch, so a recycle
costs at most one epoch.

## Failure conditions (stated before running)

For the **soft-distillation** arm, replacing the Arm A gates that stood here:

- Scale accuracy does not improve over `lora_4b_multi` (0.8183) → the discarded
  soft targets were not the binding constraint; report it and stop.
- NEI recall on single-passage rows does not move → the hardening was not the
  mechanism, regardless of what the overall accuracy does.
- Scale accuracy falls below ~0.8100, or raw ECE degrades beyond noise → the weight
  is over-hedging; sweep `soft_target_weight` down or discard.
- Any capability regression on Noul/Score or the six shapes → not shippable
  regardless of accuracy.
- The usable-soft count on the training set is far below 10,319 of 12,119 (the
  1,800 score rows legitimately gate off) → the KL path is misconfigured.
