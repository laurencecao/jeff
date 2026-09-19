"""Tests for the fail-closed train/eval disjointness gate.

Every test builds a tiny synthetic train pool + eval assets in a temp dir and
calls scripts.check_train_eval_disjoint.run_gate directly — no real data files
are touched and nothing is trained.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_clf.model import state_to_text  # noqa: E402
from scripts.check_train_eval_disjoint import run_gate  # noqa: E402

QUESTION = (
    "Fact-check the claim in `state`.\n"
    "supported: The evidence affirms the claim.\n"
    "refuted: The evidence denies the claim.\n"
    "not_enough_info: The evidence is silent."
)


def sft_row(row_id, state, *, split="train", group_id=None, gt_row_id=None):
    user = f"{QUESTION}\n\nState:\n{state_to_text(state)}"
    row = {
        "row_id": row_id,
        "split": split,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "supported"},
        ],
    }
    if group_id is not None:
        row["group_id"] = group_id
    if gt_row_id is not None:
        row["meta"] = {"gt_row_id": gt_row_id}
    return row


def eval_row(row_id, state, *, split="test", group_id=None):
    row = {"row_id": row_id, "split": split, "state": state}
    if group_id is not None:
        row["group_id"] = group_id
    return row


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class GateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gt = self.dir / "ground_truth.jsonl"
        self.other = self.dir / "eval_other.jsonl"
        self.train = self.dir / "sft_train.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, train=None, assets=None):
        if train is None:
            train = self.train
        if assets is None:
            assets = [("gt", self.gt), ("other", self.other)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_gate(train, assets)
        return code, buf.getvalue()

    def seed_clean(self):
        """A minimal disjoint world that must PASS."""
        write_jsonl(self.gt, [
            eval_row("gt-train-1", {"claim": "train claim", "evidence": [{"text": "src ev"}]},
                     split="train", group_id="g-src"),
            eval_row("gt-val-1", {"claim": "held out claim",
                                  "evidence": [{"title": "Doc", "text": "held out ev"}]},
                     split="val", group_id="g-val"),
        ])
        write_jsonl(self.other, [
            eval_row("other-1", {"claim": "another claim", "evidence": [{"text": "other ev"}]},
                     group_id="g-other"),
        ])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "train claim", "evidence": [{"text": "src ev"}]},
                    group_id="g-src", gt_row_id="gt-train-1"),
        ])

    # --- fail-closed coverage ---------------------------------------------

    def test_absent_eval_asset_fails(self):
        write_jsonl(self.gt, [eval_row("gt-val-1", {"claim": "c"}, split="val")])
        write_jsonl(self.train, [sft_row("t-1", {"claim": "x", "evidence": [{"text": "e"}]})])
        missing = self.dir / "nope.jsonl"
        code, out = self.run_gate(assets=[("gt", self.gt), ("gone", missing)])
        self.assertEqual(code, 1)
        self.assertIn("not found", out)

    def test_empty_eval_asset_fails(self):
        self.gt.write_text("")
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [sft_row("t-1", {"claim": "x", "evidence": [{"text": "e"}]})])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("empty", out)

    def test_eval_asset_with_no_protected_rows_fails(self):
        # ground_truth with only train rows: nothing held out -> cannot verify.
        write_jsonl(self.gt, [
            eval_row("gt-train-1", {"claim": "c", "evidence": [{"text": "e"}]}, split="train"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c2", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [sft_row("t-1", {"claim": "x", "evidence": [{"text": "e"}]})])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("0 protected rows", out)

    def test_zero_train_rows_fails(self):
        self.seed_clean()
        self.train.write_text("")
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("SFT pool is empty", out)

    def test_malformed_prompt_fails(self):
        self.seed_clean()
        bad = {
            "row_id": "t-bad", "split": "train",
            "messages": [{"role": "user", "content": "no state header here"}],
        }
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "ok", "evidence": [{"text": "e"}]}),
            bad,
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("unparseable", out)

    def test_missing_evidence_marker_fails(self):
        # Claim present but no "Evidence:" marker -> unparsed, not empty evidence.
        self.seed_clean()
        bad = {
            "row_id": "t-noev", "split": "train",
            "messages": [{"role": "user", "content": f"{QUESTION}\n\nState:\nClaim: orphan claim\n"}],
        }
        write_jsonl(self.train, [bad])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("unparseable", out)

    def test_nontrain_row_fails(self):
        self.seed_clean()
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "x", "evidence": [{"text": "e"}]}),
            sft_row("t-val", {"claim": "y", "evidence": [{"text": "e"}]}, split="val"),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("non-train split", out)

    def test_unknown_ground_truth_split_fails(self):
        self.seed_clean()
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "c", "evidence": [{"text": "e"}]}, split="val"),
            eval_row("gt-mystery", {"claim": "c2", "evidence": [{"text": "e"}]}, split="dev"),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("unknown/missing split", out)

    # --- leakage signatures -------------------------------------------------

    def test_exact_state_overlap_fails(self):
        state = {"claim": "shared claim", "evidence": [{"title": "T", "text": "same ev"}]}
        write_jsonl(self.gt, [eval_row("gt-val-1", state, split="val")])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [sft_row("t-1", state)])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("exact state overlap", out)

    def test_title_bearing_evidence_overlap_fails(self):
        # The regression this gate exists for: eval text-only vs SFT title:text
        # rendering must NOT hide an identical state.
        state = {"claim": "titled", "evidence": [{"title": "Some Page", "text": "body text"}]}
        write_jsonl(self.gt, [eval_row("gt-val-1", state, split="val")])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [sft_row("t-1", state)])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("exact state overlap", out)

    def test_reordered_evidence_overlap_fails(self):
        ev = [{"title": "A", "text": "first"}, {"title": "B", "text": "second"}]
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "c", "evidence": ev}, split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "c", "evidence": list(reversed(ev))}),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("reordered evidence-set overlap", out)
        self.assertNotIn("[FAIL] exact state overlap", out)

    def test_group_overlap_fails(self):
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "c", "evidence": [{"text": "e"}]},
                     split="val", group_id="shared-group"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "different", "evidence": [{"text": "different"}]},
                    group_id="shared-group"),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("group_id overlap", out)

    def test_gt_row_id_naming_protected_row_fails(self):
        write_jsonl(self.gt, [
            eval_row("gt-val-9", {"claim": "held out", "evidence": [{"text": "e"}]}, split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "different", "evidence": [{"text": "different"}]},
                    gt_row_id="gt-val-9"),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("gt_row_id names a protected eval row", out)

    # --- allowed cases --------------------------------------------------------

    def test_clean_pool_passes(self):
        self.seed_clean()
        code, out = self.run_gate()
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)
        self.assertIn("[protected]", out)

    def test_train_source_inclusion_allowed(self):
        # A train row derived from a ground_truth TRAIN row (same state, same
        # group, gt_row_id pointing at it) is legitimate provenance, not leakage.
        src = {"claim": "source claim", "evidence": [{"title": "D", "text": "source ev"}]}
        write_jsonl(self.gt, [
            eval_row("gt-train-1", src, split="train", group_id="g-src"),
            eval_row("gt-val-1", {"claim": "held", "evidence": [{"text": "h"}]}, split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", src, group_id="g-src", gt_row_id="gt-train-1"),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 0, out)

    def test_same_claim_different_evidence_allowed(self):
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "shared claim", "evidence": [{"text": "eval ev"}]},
                     split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "shared claim", "evidence": [{"text": "train ev"}]}),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 0, out)

    def test_shared_snippet_allowed(self):
        # One overlapping sentence inside different evidence sets is not leakage.
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "c1",
                                  "evidence": [{"text": "shared sentence plus eval-only"}]},
                     split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "c2",
                            "evidence": [{"text": "shared sentence plus train-only"},
                                         {"text": "extra"}]}),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 0, out)

    def test_empty_evidence_state_is_legitimate_not_unparsed(self):
        # state_to_text renders empty evidence as "(none)"; both sides must
        # parse it as a real (empty-list) state, and identical empty-evidence
        # states must still collide.
        state = {"claim": "no evidence claim", "evidence": []}
        write_jsonl(self.gt, [eval_row("gt-val-1", state, split="val")])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [sft_row("t-1", state)])
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertIn("exact state overlap", out)
        self.assertNotIn("unparseable", out)

    def test_empty_evidence_disjoint_passes(self):
        write_jsonl(self.gt, [
            eval_row("gt-val-1", {"claim": "no evidence claim", "evidence": []}, split="val"),
        ])
        write_jsonl(self.other, [eval_row("o-1", {"claim": "c", "evidence": [{"text": "e"}]})])
        write_jsonl(self.train, [
            sft_row("t-1", {"claim": "different claim", "evidence": []}),
        ])
        code, out = self.run_gate()
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main()
