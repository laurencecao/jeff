"""Static check that the PEFT wrap is single-per-path (no nested adapters).

A resume that calls PeftModel.from_pretrained on an ALREADY-wrapped model nests
adapters and mismatches optimizer/state keys. The fix is to branch BEFORE
wrapping: from_pretrained receives the raw base model, and get_peft_model runs
only on a fresh start.

This is a source-structure test, not a training run: it loads no model and does
no optimiser step, so it is safe on any box and under the all-training-on-Colab
constraint.

Run: uv run python -m scripts.check_resume_branch
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts/jev_clf_lora_train.py"


def main() -> None:
    text = SRC.read_text()
    lines = text.split("\n")
    tree = ast.parse(text)

    # Locate the two wrap calls and the branch that must separate them, using
    # AST line numbers rather than guessing at text ranges.
    fp = [n.lineno for n in ast.walk(tree)
          if isinstance(n, ast.Call) and "PeftModel" in ast.dump(n.func)
          and "from_pretrained" in ast.dump(n.func)]
    gp = [n.lineno for n in ast.walk(tree)
          if isinstance(n, ast.Call) and "get_peft_model" in ast.dump(n.func)]
    # A top-level `if resuming:` in main
    ifs = [n for n in ast.walk(tree)
           if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
           and n.test.id == "resuming"]

    print(f"from_pretrained call at line(s) : {fp}")
    print(f"get_peft_model call at line(s)  : {gp}")
    print(f"'if resuming:' node at line(s)  : {[n.lineno for n in ifs]}")

    ok = True
    if len(fp) != 1 or len(gp) != 1:
        print("FAIL - expected exactly one call of each wrap")
        ok = False
    if len(ifs) != 1:
        print("FAIL - expected exactly one `if resuming:` branch")
        ok = False
    if ok:
        node = ifs[0]
        # from_pretrained must live inside the if-branch
        body_lines = {n.lineno for n in ast.walk(node) if hasattr(n, "lineno")}
        in_if = fp[0] in body_lines and fp[0] <= max(body_lines)
        # get_peft_model must live in the ELSE branch (node.orelse)
        orelse_lines = set()
        for o in node.orelse:
            orelse_lines |= {n.lineno for n in ast.walk(o) if hasattr(n, "lineno")}
        in_else = gp[0] in orelse_lines
        print(f"from_pretrained inside the resume branch : {in_if}")
        print(f"get_peft_model inside the else branch    : {in_else}")
        if not (in_if and in_else):
            print("FAIL - the two wraps are not separated by the branch")
            ok = False

        # The base model must not be wrapped before the branch
        pre = "\n".join(lines[: node.lineno - 1])
        if "get_peft_model(" in pre:
            print("FAIL - base model is wrapped BEFORE the resume branch")
            ok = False
        else:
            print("base model unwrapped before the branch  : True")

    print()
    print("PASS - one wrap per path; from_pretrained receives the raw base model"
          if ok else "FAIL - see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
