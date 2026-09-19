# Train your own Jeff

A walkthrough for building a typed decision model on your own data.
Everything here uses the scripts in this repository; the released
`GestaltLabs/Jeff-1` adapter was trained with the same code path.

## How the model learns

Jeff is a LoRA adapter on an instruct LLM (release: Qwen3-4B-Instruct-2507).
There is no classification head. Each training example is a chat conversation
whose assistant turn is exactly the answer label. The trainer masks every
prompt and padding token and supervises **only the answer tokens** — so the
model is trained to put probability mass on the label tokens given the prompt.

At inference the label distribution is read from the model's own token
probabilities over the candidate labels (`jev_clf/readout.py`). Because label
names and descriptions live in the prompt, a new label set works without
retraining.

## Data format

Training rows are JSONL. One row:

```json
{
  "row_id": "my-data-0001",
  "question_id": "verdict",
  "split": "train",
  "messages": [
    {"role": "system", "content": "You are a fact-checking classifier. The user gives you a question, a list of answer labels with their definitions, and a state containing a claim and evidence passages. Answer with exactly one of the listed label names and nothing else."},
    {"role": "user", "content": "Fact-check the claim using the evidence.\n\nClaim: The museum opened in 1998.\n\nEvidence:\n1. The museum first opened to visitors in 1998.\n\nLabels:\n- supported: The evidence establishes the claim.\n- refuted: The evidence contradicts the claim.\n- not_enough_info: The evidence is insufficient to decide.\n\nVerdict:"},
    {"role": "assistant", "content": "supported"}
  ],
  "label": "supported",
  "label_space": ["supported", "refuted", "not_enough_info"],
  "schema_id": "fc-my-0",
  "source": "my_dataset",
  "weight": 1.0
}
```

Rules enforced by the trainer (fail closed):

- Every row must carry `"split": "train"`. Validation rows go in a separate
  file and must not appear in the training file.
- The assistant turn is the label. Keep `prompt_format: eval` in the config so
  your training prompt is rendered exactly like the evaluation harness
  (label definitions, then `Verdict:`), with the trainer adding the leading
  space before the label.
- Give rows from one source/creation batch a shared `group_id`-style
  identifier in `meta` where possible, and keep any near-duplicate text out of
  more than one split. Leakage inflates validation scores.

Write a small generation script for your data rather than hand-writing JSONL:
build the user message from your claim/evidence and label definitions, then
emit one row per question. The same state can carry several question types
(Choice, yes/no, ordered Score levels) — that is how the released adapter
learned all three primitives.

## Configuration

A starter config ships as `configs/my_jeff.yaml` (modeled on
`configs/jev_clf_lora_multi.yaml`, which trained the released adapter):

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
dtype: bfloat16
max_length: 2048
gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

optim:
  lr: 0.0001
  weight_decay: 0.0
  epochs: 2
  batch_size: 4          # 4B: keep vocab-sized logits inside ~40 GB
  grad_accum: 8          # effective batch = 32
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  seed: 42

data:
  train: data/my_task/sft_train.jsonl
  val: data/my_task/sft_val.jsonl
  prompt_format: eval
  completion_leading_space: true

out_dir: artifacts/jev_clf/my_jeff
```

MLflow logging in the trainer is best-effort: if no tracker is running it
prints `[mlflow] skipped` and continues. You can delete the `mlflow:` block.

## Smoke test, then real training

Smoke test locally (CPU, tiny — just proves the plumbing):

```bash
uv run python -m scripts.jev_clf_lora_train \
  --config configs/my_jeff.yaml \
  --max-rows 16 --epochs 1 --out-dir /tmp/jeff_smoke
```

Real training needs a CUDA GPU (a Colab T4 is enough for a small dataset;
the release run used an A100 with the batch sizes above). The trainer will
fall back to CPU/MPS without complaint, which is far too slow to be useful —
run the real job on Colab.

If you train on Colab, remember sessions are reclaimed and **lose `/content`**:

- upload a tarball of the repo, extract, then launch training detached
  (`nohup ... > run.log 2>&1 &`);
- make the launcher pack the finished adapter into a tarball immediately on
  success and `stat` its size;
- download that tarball as soon as it appears, and verify byte counts before
  extracting. A truncated download fails `tar tzf` — re-download, do not trust it.

## Evaluate

Evaluation uses `DecisionRow` JSONL (a different format from training rows):
one row per state, with `questions` and gold `labels` per question.

```json
{
  "row_id": "my-eval-0001",
  "source": "my_dataset",
  "split": "val",
  "group_id": "g1",
  "state": {"claim": "The museum opened in 1998.",
             "evidence": ["The museum first opened to visitors in 1998."]},
  "questions": {"verdict": {"kind": "choice",
     "instructions": "Judge the claim using only the supplied evidence.",
     "criteria": {"supported": "The evidence establishes the claim.",
                  "refuted": "The evidence contradicts the claim.",
                  "not_enough_info": "The evidence is insufficient."}}},
  "labels": {"verdict": {"supported": 1.0, "refuted": 0.0, "not_enough_info": 0.0}},
  "label_source": "human"
}
```

```bash
uv run python -m scripts.jev_clf_lm_eval \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter artifacts/jev_clf/my_jeff \
  --data-file data/my_task/eval_val.jsonl
```

Without `--data-file` the harness scores the repository's default
fact-check split (`--split val` on `ground_truth.jsonl`), which is the
wrong yardstick for a task-specific model — always pass your own file.

Score accuracy against **human labels**, not against the teacher or data
source you trained from. If you distilled examples from another model, report
agreement with that model as a separate number — agreement can be high while
accuracy is not.

## Use your adapter

```python
from jev_clf.client import SystemOneClient, Choice

client = SystemOneClient(adapter="artifacts/jev_clf/my_jeff")
result = client.system_one(state, {"verdict": Choice(instructions=..., criteria=...)})
```

## Notes and limits

- The examples above show the format; they are far too few to train a useful
  model. Expect to need thousands of rows representative of your task, and a
  validation set your training never saw.
- `prompt_format: eval` keeps training and evaluation prompts identical; if
  you change the system prompt or label wording, change it in both places.
- The released adapter's headline numbers came from a specific 9,730-row
  fact-check evaluation; they do not transfer to a new task or dataset. Your
  model's quality is whatever your own held-out evaluation says it is.
- Details of the release run (splits, metrics, provenance) are in
  `RELEASE_JEFF1.md` and `MODEL_CARD_JEFF1.md`.