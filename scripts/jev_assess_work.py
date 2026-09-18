"""Put Jev to work as a calibrated judge over our own work.

Uses the TypeSafe API directly (the installed `typesafe-ai` skill is docs/API
guidance, not an MCP server — verify with `grep -c mcp SKILL.md`).

Following the skill's guidance:
  * one narrow, coherent judgment per question, several asked in ONE call
  * question IDs are code-side only; full meaning lives in instructions/criteria
  * label sets and their definitions are declared at call time (text-conditioned)
  * results are probabilities, not prose — code decides what to do with them

Two things are assessed:
  A. The Corrective Ornstein method (the user's prior work, using an
     Ornstein-Uhlenbeck / drift-diffusion framing scored by AUC) — how much of
     it transfers to this task.
  B. Our own jeff state of evidence — which lever is most likely to close
     the remaining gap, and whether our headline claims are supported.

    uv run python -m scripts.jev_assess_work
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "jev-1.13.0"

# --- the state: everything the judge needs, in one structured object ---------

ORSTEIN_SUMMARY = {
    "what_it_is": "Corrective Ornstein: a post-training path that turns a reflection pattern into an evidence-gated habit.",
    "core_finding": "The behavioral update is preference training (DPO) over decision-level pairs, NOT another full SFT pass.",
    "policy_states": [
        "pre-action: check a concrete risk vs proceeding without checking",
        "verification: call the real verifier vs declaring Done",
        "repair: change the failed candidate and cite the failed check vs retry unchanged",
        "escalation: search after a failed local repair vs guess from memory",
        "termination: return the receipt-bound answer after a pass vs verifying forever",
    ],
    "authorization_rule": "DONE is authorized by the compiler, not by a sentence in the teacher's reasoning. A passing receipt must have evidence_type test_result/tool_result/execution; an LLM judge cannot certify that code ran.",
    "quantitative_parts": {
        "drift_diffusion": "Drift Diffusion Model (DDM) evidence accumulation over reasoning traces: per-segment evidence trajectories, final evidence value, curriculum bins, per-step loss weights.",
        "auc": "An audit ROC-AUC separating accepted from rejected pipeline behavior, described by the author as a STRUCTURAL diagnostic, explicitly not a substitute for held-out human semantic labels.",
        "uhlenbeck": "The Ornstein-Uhlenbeck framing is used for conditional length / position / right-tail SHAPE of reasoning traces; profiles keep raw empirical donor vectors rather than fitting a Gaussian.",
    },
    "guards": ["immutable SHA-256 plan ids", "fail-closed resume", "transitive grouping so duplicate problems cannot cross splits"],
}

OUR_WORK = {
    "task": "Given a claim and its retrieved evidence passages, output a calibrated distribution over supported / refuted / not_enough_info. Text-conditioned: the label set and each label's definition arrive in the prompt at call time.",
    "model": "LoRA adapter (4.36M params, 0.28%) on Qwen2.5-1.5B-Instruct; classification read from the model's own next-token distribution over label tokens, one forward pass, no generation loop.",
    "measured": {
        "ours_test_n199": {"accuracy": 0.759, "macro_f1": 0.738, "ece": 0.111},
        "live_jev_test_n199": {"accuracy": 0.799, "macro_f1": 0.791, "ece": 0.114},
        "zero_shot_nli_test_n199": {"accuracy": 0.668, "macro_f1": 0.658, "ece": 0.277},
    },
    "uncertainty": "n=199 gives a 95% CI of roughly +/- 0.056; ours [0.695, 0.813] and Jev [0.738, 0.849] overlap almost entirely, so 'ours is 4 points behind Jev' is NOT statistically resolvable at this sample size.",
    "training_data": "9119 examples: 7000 labelled by live Jev (soft distributions), 2119 human-labelled (FEVER / VitaminC / SciFact / Climate-FEVER).",
    "error_structure_test": "23 of 48 errors are false 'supported': 13 where truth was not_enough_info, 10 where truth was refuted. Recall 0.844 supported / 0.707 refuted / 0.644 not_enough_info.",
    "three_negatives_already_tested": [
        "per-label prior bias: identity is optimal, every other setting loses a row, so the over-claiming errors are confidently wrong and not a threshold artifact",
        "schema-wording TTA (average over wordings): accuracy -3 rows and agreement collapsed 0.769 -> 0.374",
        "human-label upweighting 3x on retrain: LM loss improved but task accuracy -1 row (within noise)",
    ],
    "teacher_quality": "Live Jev is only ~0.80 accurate on this data, so 7000 of 9119 training targets imitate a teacher wrong about one case in five.",
    "constraint": "The test split is sealed; the loop optimizes val accuracy only.",
    "no_tool_calls": "This task has no verifier, no web search, no code execution. A single forward pass produces the decision; there is nothing to repair or escalate.",
}


def main() -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set")

    state = {"orstein_method": ORSTEIN_SUMMARY, "our_work": OUR_WORK}
    client = TypeSafeClient()

    questions = {
        # A. Ornstein transfer
        "auth_rule_transfers": Noul(
            instructions=(
                "Read `orstein_method.authorization_rule` and `our_work`. The "
                "authorization rule says a completion claim must be authorized by "
                "external evidence, not by the model's own text. Our task is a "
                "single-pass classifier: no tools, no verifier, no repair loop. "
                "Does that authorization principle still transfer to our task?"
            ),
            criteria={
                "yes": "The principle transfers: it applies to what we treat as authoritative labels or answers.",
                "no": "The principle does not transfer: without tools or a verifier there is nothing to authorize.",
            },
        ),
        "ddm_transfers": Score(
            instructions=(
                "Read `orstein_method.quantitative_parts.drift_diffusion` and "
                "`our_work`. How well does the drift-diffusion evidence-accumulation "
                "machinery transfer to our model? It models the shape of a generated "
                "reasoning trace; our model emits no trace at all."
            ),
            criteria=[
                "No transfer: there is no trace to accumulate evidence over",
                "Weak: only as a loose analogy for how evidence in the passages should weigh",
                "Partial: some machinery (evidence weighting) could be recast for our inputs",
                "Strong: it applies largely as-is to our task",
            ],
        ),
        "auc_transfers": Noul(
            instructions=(
                "Read `orstein_method.quantitative_parts.auc` and `our_work.measured`. "
                "That AUC is a structural diagnostic separating accepted from rejected "
                "behavior, not a human-label quality measure. In our work we found "
                "agreement-with-teacher and accuracy-vs-human-labels diverge sharply "
                "(the 0.5B model reached 0.917 agreement at 0.458 accuracy). Should we "
                "adopt a structural accepted/rejected AUC of the same kind as an "
                "additional diagnostic for our pipeline?"
            ),
        ),
        # B. our claims and next lever
        "headline_supported": Noul(
            instructions=(
                "Read `our_work.measured` and `our_work.uncertainty`. We want to state "
                "that our model is roughly as good as live Jev at fact checking. Does "
                "the evidence provided support that statement at the stated sample size?"
            ),
            criteria={
                "yes": "Supported: the difference is small enough relative to the evidence that the claim is fair.",
                "no": "Not supported: the sample size cannot resolve the difference, so the claim should be softened.",
            },
        ),
        "next_lever": Choice(
            instructions=(
                "Read all of `our_work`, including the three negatives already tested "
                "and the error structure. We must pick ONE next lever on a ~2-row gap. "
                "Which is most likely to produce a real improvement?"
            ),
            criteria={
                "evidence_gated_labels": (
                    "Apply the Ornstein authorization rule to our data: only treat a "
                    "teacher label as authoritative when it can be checked, and drop or "
                    "demote unverifiable teacher labels."
                ),
                "dpo_decision_pairs": (
                    "Preference training over decision-level pairs (our own wrong verdict "
                    "vs the correct one on the same state), instead of more SFT."
                ),
                "more_human_data": "Scale the human-labelled data (currently 1987 rows) substantially.",
                "bigger_model": "Train the same recipe on a larger base model.",
                "abstain_engineering": (
                    "Work the not_enough_info class specifically, since it has the worst "
                    "recall and the false-supported errors concentrate there."
                ),
            },
        ),
        "overclaim_fix": Noul(
            instructions=(
                "Read `our_work.error_structure_test` and the three negatives. Our model "
                "over-claims support. Given that a post-hoc prior correction, wording "
                "averaging, and reweighting human labels all failed to fix it, is the "
                "most likely cause that the model has not learned to detect the ABSENCE "
                "of supporting evidence, rather than a decision-threshold problem?"
            ),
        ),
        "honest_report": Score(
            instructions=(
                "Read `our_work`. How strong is our overall evidence that the approach "
                "(a real language model read out as a text-conditioned classifier) is "
                "the right architecture for this task, independent of whether it yet "
                "matches Jev?"
            ),
            criteria=[
                "Weak: no clear evidence either way",
                "Moderate: architecture works, but confounded by sample size and teacher quality",
                "Strong: the step change over the frozen-encoder baseline is clear and well supported",
            ],
        ),
    }

    result = client.system_one(state, questions)

    print("=== A. Corrective Ornstein transfer ===")
    print("  authorization rule transfers        :", round(result.nouls["auth_rule_transfers"].noul, 3))
    print("  drift-diffusion transfer (score)    :", result.scores["ddm_transfers"].score,
          {k: round(v, 3) for k, v in result.scores["ddm_transfers"].probabilities.items()})
    print("  structural accepted/rejected AUC?   :", round(result.nouls["auc_transfers"].noul, 3))
    print()
    print("=== B. Our work ===")
    print("  'as good as Jev' supported?         :", round(result.nouls["headline_supported"].noul, 3))
    print("  most promising next lever           :", result.choices["next_lever"].choice)
    print("    distribution                      :", {k: round(v, 3) for k, v in result.choices["next_lever"].probabilities.items()})
    print("  absence-of-evidence is the cause?   :", round(result.nouls["overclaim_fix"].noul, 3))
    print("  architecture evidence (score)       :", result.scores["honest_report"].score,
          {k: round(v, 3) for k, v in result.scores["honest_report"].probabilities.items()})
    print()
    print("usage:", result.usage)

    dest = ROOT / "results" / "jev_assessment.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "model": result.model,
                "state": state,
                "answers": {
                    "auth_rule_transfers": result.nouls["auth_rule_transfers"].noul,
                    "ddm_transfers": {
                        "score": result.scores["ddm_transfers"].score,
                        "probabilities": result.scores["ddm_transfers"].probabilities,
                    },
                    "auc_transfers": result.nouls["auc_transfers"].noul,
                    "headline_supported": result.nouls["headline_supported"].noul,
                    "next_lever": {
                        "choice": result.choices["next_lever"].choice,
                        "probabilities": result.choices["next_lever"].probabilities,
                    },
                    "overclaim_fix": result.nouls["overclaim_fix"].noul,
                    "honest_report": {
                        "score": result.scores["honest_report"].score,
                        "probabilities": result.scores["honest_report"].probabilities,
                    },
                },
                "usage": result.usage,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("wrote", dest)


if __name__ == "__main__":
    main()
