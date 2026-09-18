"""Probe: does the `sequence` readout fix the dead-uniform Score output?

first_token picks variants[label][0]; for digit labels ("0".."3") the leading
space makes every label's first token the SAME bare-space token (220), so all
four logits are identical and the softmax is exactly uniform. The sequence
readout scores the whole label token sequence, which is distinct per label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_clf import schema as S  # noqa: E402
from jev_clf.model import state_to_text  # noqa: E402
from scripts.jev_clf_lm_eval import build_inputs, label_variants  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = str(ROOT / "artifacts/jev_clf/lora_4b")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

STATE = {
    "claim": "The Eiffel Tower is located in Barcelona.",
    "evidence": [
        {
            "title": "evidence 1",
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France.",
        }
    ],
}

SCORE_Q = S.ScoreQuestion(
    instructions="How much evidence is provided for the claim?",
    criteria=[
        "no evidence at all",
        "a single weak or unrelated passage",
        "one relevant passage",
        "several relevant passages",
    ],
)

print(f"[probe] device={DEVICE} loading {BASE} + {Path(ADAPTER).name}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.to(DEVICE).eval()
print("[probe] model loaded", flush=True)

labels = S.label_space(SCORE_Q)
variants = label_variants(tok, labels)
text = build_inputs(tok, STATE, SCORE_Q)
enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
with torch.no_grad():
    logits = model(**enc).logits[0, -1].float()

print("\n[probe] tokenisation of each label")
for lab in labels:
    ids = variants[lab]
    print(f"  {lab!r:6} ids={ids} toks={tok.convert_ids_to_tokens(ids)}")

print("\n[probe] FIRST-TOKEN readout (current behaviour)")
ids0 = [variants[lab][0] for lab in labels]
raw0 = [float(logits[i]) for i in ids0]
p0 = torch.softmax(torch.tensor(raw0), dim=-1)
for lab, i, r, p in zip(labels, ids0, raw0, p0.tolist()):
    print(f"  {lab}  token={i}  logit={r:+.4f}  p={p:.6f}")
print(f"  -> uniform? {len(set(round(x, 6) for x in p0.tolist())) == 1}")

print("\n[probe] SEQUENCE readout (candidate fix)")
scores = []
for lab in labels:
    seq = variants[lab]
    full = torch.tensor([enc["input_ids"][0].tolist() + seq], device=DEVICE)
    with torch.no_grad():
        out = model(input_ids=full).logits[0]
    lp = 0.0
    for k, tid in enumerate(seq):
        pos = full.shape[1] - len(seq) + k - 1
        lp += float(torch.log_softmax(out[pos].float(), dim=-1)[tid])
    scores.append(lp)
ps = torch.softmax(torch.tensor(scores), dim=-1)
for lab, s, p in zip(labels, scores, ps.tolist()):
    print(f"  {lab}  logprob_sum={s:+.4f}  p={p:.6f}")
print(f"  -> uniform? {len(set(round(x, 6) for x in ps.tolist())) == 1}")
exp = sum(int(l) * p for l, p in zip(labels, ps.tolist()))
print(f"\n[probe] sequence score = {exp:.4f}  (first_token score = 1.5, dead)")

print("\n[probe] DONE", flush=True)
