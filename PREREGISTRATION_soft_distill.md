# Pre-registration: soft-target distillation arm (lora_4b_soft)

Written BEFORE scoring the arm, so the prediction cannot be reverse-fitted to
whatever number comes out. Commit this claim now; check it when the A/B lands.

## Arm
- Config `configs/jev_clf_lora_soft.yaml` — byte-identical to
  `configs/jev_clf_lora_multi.yaml` except `out_dir`, `mlflow.run_name`, and
  `optim.soft_target_weight: 1.0`. One variable.
- Loss becomes `CE + 1.0 * mean_rows KL(teacher || student)` at the first
  supervised position, reading student mass off the first token of each label.
- Baseline to beat: `lora_4b_multi`, scale accuracy **0.8183** (n=9730).
- Comparison is on the scale split, paired (same rows), by paired McNemar.
  The 199-row val split is NOT a valid decision instrument: paired p=0.59,
  bootstrap CI ±5.6 points. Ignore val for the keep/discard call.

## Why this should work
`masked_lm_loss` was hard-label token CE only. All 12,119 training rows carry
`target_probs` (the teacher's full distribution), and the pipeline hardened each
to its argmax and never used the distribution. The teacher's hedging was
discarded exactly where our model is wrong.

Measured signal that was being thrown away, on the 9,119 Choice training rows:
- 2,591 rows (28.4%) have a runner-up class >= 0.05.
- Hedged rate by gold label: NEI 32.8% > refuted 28.9% > supported 20.8%.
- On the 3,252 NEI-labelled rows: 1,065 (32.7%) have teacher P(NEI) < 0.95,
  and 569 have P(NEI) < 0.80 — the teacher genuinely hedging on NEI.

## PRE-REGISTERED PREDICTION (falsifiable)
1. **Direction**: scale accuracy IMPROVES vs 0.8183. Magnitude expected small:
   **+10 to +40 rows** (0.10 to 0.41 accuracy points). I do NOT expect the full
   +311-row oracle.
2. **Where**: the gain concentrates in `not_enough_info` recall, specifically on
   single-passage rows (baseline NEI recall there 0.5229).
3. **Where NOT**: `supported` recall should NOT improve much — we are already
   above Jev there (0.9469 vs 0.9339), and the soft targets mostly add hedging,
   which if anything trades a little supported precision for NEI recall.
4. **Calibration**: scale ECE should change by less than 0.01 either way. Soft
   targets sharpen the boundary; they do not obviously recalibrate the bulk.
5. **Significance**: I predict the improvement is NOT statistically significant
   on its own (paired p > 0.05), because +10..40 rows against the measured
   discordant-pair structure gives z well under 2. If it does clear p<0.05,
   that is a genuine surprise worth stating as such.

## Counter-evidence I already have (why the effect may be small)
Our errors are NOT mostly at the boundary. On the 513 scale rows where gold is
`not_enough_info` and we answer `supported`, our own P(NEI) is mean 0.146,
median 0.087 — we are confidently wrong, not marginally wrong. Only 134 of those
rows have P(NEI) >= 0.25, and only 38 have >= 0.40. A KL term can only pull rows
where the student already puts *some* mass on the right class. So the reachable
subset is a minority of the deficit.

## Refutation conditions (any of these kills the arm)
- Scale accuracy <= 0.8183 (no gain): arm is not worth keeping as the default.
- NEI recall on 1-passage rows does NOT move (prediction 2 fails): then the
  hardening was not the binding mechanism and the loss change is not the lever.
- Scale accuracy drops below ~0.8100: the KL is over-hedging and damaging the
  supported class; discard.
- NaN/inf in the loss, or the soft count printing far below ~10,300 of 12,119
  (the 1,800 score rows legitimately gate off — they share first token 220).

## Protocol integrity notes
- Choice readout path must remain bit-unchanged for the baseline arm, so the
  comparison isolates training. Verify `lora_4b_multi` still reproduces 0.8183
  if anything in the shared code path was touched.
- Never use the 199-row val split to pick between arms.
- Report BOTH scale accuracy and scale ECE; do not quote only the favourable one.
