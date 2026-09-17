# jev_clf — cross-slice contracts

Our own decision-only fact-checking model. Frozen interfaces live in
`jev_clf/schema.py`; this file is the prose half of that contract. Scripts
resolve paths from the repo root (`Path(__file__).resolve().parents[1]`).

## The task

Text-conditioned classification: a request carries `state` plus one or more
typed questions whose answer space is *named in natural language in the
request*. The model returns a probability distribution over that answer space.
Labels are not fixed by training — `ChoiceQuestion.criteria` keys are the
labels and its values are their definitions.

Primary task (`jev_clf.schema.make_factcheck_choice`): given a claim and the
evidence passages retrieved for it, return a calibrated distribution over
`supported` / `refuted` / `not_enough_info`. Evidence-zero rows must answer
`not_enough_info` — the model refuses rather than answering from world
knowledge.

## Data files

| File | Rows | Written by | Read by |
|---|---|---|---|
| `data/factcheck/fixture.jsonl` | `DecisionRow` | committed by hand | 03, 04, 06 |
| `data/factcheck/distill_pilot.jsonl` | `DecisionRow` | `gen` | 04, 06 |
| `data/factcheck/distill_full.jsonl` | `DecisionRow` | `gen` | 04, 06 |
| `data/factcheck/ground_truth.jsonl` | `DecisionRow` | `data` | 04, 06 |
| `data/factcheck/preds_<model>.jsonl` | `PredictionRow` | `eval`, `baselines` | 06 |
| `data/factcheck/jev_cache.jsonl` | raw teacher responses | `jev` | — |

Every file is JSONL, one schema object per line. `split` is `train|val|test`
and is assigned by the writer; rows sharing a `group_id` never straddle splits.

## Module APIs

### `jev_clf.schema`  (frozen — written already, do not fork)
```python
Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="kind")]
label_space(q) -> list[str]
question_to_text(q) -> str                 # the model's text view of a question
DecisionRow / PredictionRow                # pydantic models
read_rows / write_rows / append_rows
read_predictions / write_predictions
normalize(probs) / argmax_label(probs) / one_hot(labels, winner)
to_option_scorer_rows(rows) -> list[dict]  # lossy adapter for jevlike/jevbetter
make_factcheck_choice(...) / make_factcheck_noul(...)
FACTCHECK_QUESTION_ID, FACTCHECK_LABELS
```

### `jev_clf.jev`
```python
class JevTeacher:
    def __init__(self, model: str = "jev-1.13.0", cache_path: str | Path = JE_CACHE,
                 api_key: str | None = None, max_retries: int = 4)
    def ask(self, state, questions: Questions) -> TeacherAnswer
    def ask_rows(self, rows: Iterable[DecisionRow]) -> Iterator[DecisionRow]  # fills labels
    def spent_requests(self) -> int
class TeacherAnswer:  # distributions, confidence, usage, latency_ms, model
```
Resumable: cache key = hash(state, questions, model). A cached call never
re-hits the network. Rate limits: 1200 req/min, 250k tok/s; SDK retries.

### `jev_clf.gen`
```python
def generate_rows(n: int, seed: int = 42, teacher: JevTeacher | None = None,
                  domains: list[str] | None = None,
                  only: str | None = None) -> Iterator[DecisionRow]
def instruction_variants(kind: str = "factcheck", seed: int = 42) -> list[str]
```
`only="synthetic"` yields oracle-labelled synthetic rows with no network call;
`only="jev"` labels them with the teacher. Labels for Jev-distilled rows are Jev's full
distribution (`label_source="jev-1.13.0"`), not an argmax — soft targets are
what carry calibration.

### `jev_clf.data`
```python
def load_ground_truth(sources=("fever","vitaminc","scifact"), max_per_source=2000,
                      seed=42) -> list[DecisionRow]
def assign_splits(rows, ratios=(0.8,0.1,0.1), seed=42) -> list[DecisionRow]  # grouped
```
Loaders must produce `label_source="ground_truth"` rows in the frozen format
and cache a local copy under `data/factcheck/raw/`. Never download at eval time.

### `jev_clf.model`
```python
class OptionScorer:
    def __init__(self, encoder_name=..., head_width=..., freeze_encoder=True)
    def forward(self, states, questions) -> list[dict[str, float]]  # probs per option
    def save(self, dir) / classmethod load(cls, dir)
```
One pass per decision: the state is encoded once, each option's
(label + definition + instructions) becomes a query vector, options attend to
the context, one score each, softmax across options.

### `jev_clf.train`
```python
def train(rows, cfg: dict, out_dir) -> dict   # returns metrics + artifact paths
```
Soft-target cross-entropy against `row.targets(qid)`, weighted by `row.weight`.
Calibration (temperature scaling; isotonic as a comparison) is fit on `val`.

### `jev_clf.eval`
```python
def agreement(rows, preds) -> dict            # vs Jev, on held-out schemas
def ground_truth_metrics(rows, preds) -> dict # accuracy, macro-F1, ECE, Brier
def reliability_bins(rows, preds, n_bins=10) -> list[dict]
def jaggedness_suite() -> list[DecisionRow]   # counting, numeric, indirection, adversarial
def throughput(model, rows) -> dict
```
`agreement` measures cloning fidelity. `ground_truth_metrics` measures quality.
They are reported separately and never averaged — agreement alone is circular.

### `jev_clf.baselines`
```python
class JevClassifier      # live Jev answering the same rows (upper bound)
class ZeroShotNLI        # DeBERTa-MNLI entailment scoring
class OptionScorerBase   # jevlike/jevbetter via to_option_scorer_rows (lossy)
```

## Configs

- `configs/jev_clf.yaml` — encoder, head width, lr, epochs, batch size, seeds,
  calibration method, weight of each data source, split ratios.

## Conventions

- Seeds come from config; default 42.
- No API keys in files. `TYPESAFE_API_KEY` from env.
- Run with `uv run` from the repo root (`from jev_clf.schema import ...`).
- Every score/CLI writes into `artifacts/jev_clf/` and logs to MLflow
  (`mlflow.set_tracking_uri("http://127.0.0.1:5001")`), experiment `jev-clf`.
- Deterministic given a seed; state machine + device in every report.
