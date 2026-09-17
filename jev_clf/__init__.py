"""jev_clf — our own decision-only fact-checking model.

Slices:
    schema   frozen data/question contract (import this, do not fork it)
    jev      live Jev teacher client, resumable
    gen      programmatic question-schema + claim generator
    data     real fact-verification datasets -> DecisionRow
    model    the text-conditioned option scorer
    train    training loop + calibration
    eval     agreement, ground-truth metrics, ECE, jaggedness suite
    baselines  Jev-as-classifier, zero-shot NLI, open option scorers

Run everything from the repo root, e.g.:
    uv run python -m jev_clf.train --config configs/jev_clf.yaml
"""

__version__ = "0.1.0"
