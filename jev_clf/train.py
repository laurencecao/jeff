"""Soft-target training + calibration for the jev_clf OptionScorer.

Loss is soft-target cross-entropy (KL up to a constant) against
``row.targets(qid)`` — the teacher's full distribution, never hardened to an
argmax — summed over every question in the row and multiplied by the row's
weight (``row.weight`` x ``cfg["source_weights"][row.source]``).

Calibration is fit on the val split: temperature scaling is the method that
ships inside the checkpoint (``model.temperature`` is applied in
``forward``); isotonic regression is fit as a comparison and stored in
``calibration.json``. Val ECE is reported before and after each method.

MLflow logging is guarded: the tracking server may be down, and a failed
MLflow call must never fail a run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .model import OptionScorer
from .schema import DecisionRow, read_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts" / "jev_clf"
MLFLOW_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT = "jev-clf"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def resolve_device(cfg: dict) -> torch.device:
    """Device from cfg, or auto: CUDA if available, else MPS, else CPU.

    The previous version only knew about MPS and silently picked CPU on a CUDA
    box, which made an A100 run spin the trainer on CPU at ~600 cpu-seconds per
    20 wall-seconds with the GPU at 0%.
    """
    want = cfg.get("device", "auto")
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_weight(row: DecisionRow, cfg: dict) -> float:
    """``row.weight`` x the per-source multiplier from cfg."""
    return float(row.weight) * float(
        cfg.get("source_weights", {}).get(row.source, 1.0)
    )


def soft_ce(
    logits: torch.Tensor, target: torch.Tensor, weight: float
) -> torch.Tensor:
    """Weighted soft-target cross-entropy for one (row, question) pair."""
    logp = torch.log_softmax(logits, dim=-1)
    return -(target * logp).sum() * weight


def _target_tensor(labels: list[str], target: dict[str, float]) -> torch.Tensor:
    return torch.tensor([target[l] for l in labels], dtype=torch.float32)


def batch_loss(
    model: OptionScorer,
    rows: list[DecisionRow],
    cfg: dict,
) -> tuple[torch.Tensor, int]:
    """MEAN weighted soft-CE per (row, question) pair in the batch.

    Returns (mean_loss, n_pairs). Weighting is inside the mean.

    NOTE: this returned a SUM over pairs until it was found to be the cause of
    `val_loss=nan` on Colab — the reported value scaled with the number of pairs
    (x20 on 20 val rows, and past float range on 199), which looked like a
    numerical overflow but was a sum-vs-mean accounting bug.
    """
    logits = model.forward_logits(
        [r.state for r in rows], [r.questions for r in rows]
    )
    total = torch.zeros((), device=model.device)
    for pair in logits:
        row = rows[pair["row_index"]]
        target = _target_tensor(
            pair["labels"], row.targets(pair["question_id"])
        ).to(model.device)
        total = total + soft_ce(pair["logits"], target, row_weight(row, cfg))
    return total / max(len(logits), 1), len(logits)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on soft targets.

    Bins by predicted confidence (max prob); accuracy is the mean target
    probability assigned to the predicted label — the soft-target analogue of
    correctness.
    """
    conf = probs.max(axis=1)
    acc = targets[np.arange(len(probs)), probs.argmax(axis=1)]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def _collect(
    model: OptionScorer, rows: list[DecisionRow]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(probs [N,Kmax], targets [N,Kmax], qids) over all val pairs.

    K varies per question; rows are padded with zeros to the widest label
    space seen. ``qids`` lets the isotonic fit group by question id.
    """
    logits = model.forward_logits(
        [r.state for r in rows], [r.questions for r in rows]
    )
    k_max = max(len(p["labels"]) for p in logits)
    probs = np.zeros((len(logits), k_max))
    tgts = np.zeros((len(logits), k_max))
    qids = []
    for i, pair in enumerate(logits):
        p = torch.softmax(pair["logits"], dim=-1).detach().cpu().numpy()
        t = _target_tensor(
            pair["labels"],
            rows[pair["row_index"]].targets(pair["question_id"]),
        ).numpy()
        probs[i, : len(p)] = p
        tgts[i, : len(t)] = t
        qids.append(pair["question_id"])
    return probs, tgts, qids


def fit_temperature(
    model: OptionScorer, rows: list[DecisionRow], max_iter: int = 50
) -> float:
    """Scalar temperature minimizing soft-target CE on val. 1.0 if no val."""
    if not rows:
        return 1.0
    logits = model.forward_logits(
        [r.state for r in rows], [r.questions for r in rows]
    )
    logit_list, tgt_list = [], []
    for pair in logits:
        logit_list.append(pair["logits"].detach())
        tgt_list.append(
            _target_tensor(
                pair["labels"],
                rows[pair["row_index"]].targets(pair["question_id"]),
            ).to(model.device)
        )
    log_t = torch.zeros((), device=model.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        t = log_t.exp()
        loss = sum(
            soft_ce(lg / t, tg, 1.0) for lg, tg in zip(logit_list, tgt_list)
        )
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def fit_isotonic(
    probs: np.ndarray, targets: np.ndarray, qids: list[str]
) -> dict[str, Any]:
    """Per-question isotonic regression, pred-prob -> target-prob.

    Fit on flattened (predicted, target) pairs per question id. Degenerate
    fits (constant predictions, <3 points) fall back to identity.
    """
    from sklearn.isotonic import IsotonicRegression

    params: dict[str, Any] = {}
    for qid in sorted(set(qids)):
        idx = [i for i, q in enumerate(qids) if q == qid]
        x = probs[idx].ravel()
        y = targets[idx].ravel()
        keep = x > 0  # drop padding positions
        x, y = x[keep], y[keep]
        if len(x) < 3 or np.all(x == x[0]):
            params[qid] = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        params[qid] = {
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
        }
    return params


def apply_isotonic(
    probs: np.ndarray, qids: list[str], params: dict[str, Any]
) -> np.ndarray:
    """Apply fitted isotonic maps and renormalize each row to sum to 1."""
    out = np.zeros_like(probs)
    for i, qid in enumerate(qids):
        p = params.get(qid)
        row = probs[i]
        mask = row > 0
        if p is not None:
            row = row.copy()
            row[mask] = np.interp(row[mask], p["x"], p["y"])
        total = row.sum()
        out[i] = row / total if total > 0 else mask / mask.sum()
    return out


# ---------------------------------------------------------------------------
# MLflow — guarded; a dead server never fails a run
# ---------------------------------------------------------------------------

def _mlflow_log(fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — logging must never fail training
        print(f"[mlflow] skipped: {e}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(rows: list[DecisionRow], cfg: dict, out_dir: str | Path) -> dict:
    """Train the OptionScorer head and calibrate on val.

    ``rows`` carry their own ``split``; ``cfg`` is the parsed
    ``configs/jev_clf.yaml`` dict. Returns metrics + artifact paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(cfg)

    model = OptionScorer(
        encoder_name=cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
        head_width=int(cfg.get("head_width", 256)),
        heads=int(cfg.get("heads", 4)),
        freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        max_length=int(cfg.get("max_length", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    if not train_rows:
        raise ValueError("no train rows")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 8))
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch_size))
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    patience = int(cfg.get("patience", 5))
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: list[dict] = []
    rng = random.Random(seed)
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        order = train_rows[:]
        rng.shuffle(order)
        epoch_loss = 0.0
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            loss, _ = batch_loss(model, batch, cfg)  # already a mean per pair
            loss = loss / len(batch)  # scale for gradient accumulation
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            epoch_loss += float(loss.detach()) * len(batch)
        epoch_loss /= len(train_rows)

        val_loss = float("nan")
        if val_rows:
            model.eval()
            with torch.no_grad():
                vl, _ = batch_loss(model, val_rows, cfg)  # already a mean per pair
                val_loss = float(vl)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"epoch {epoch}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")


        if val_rows:
            if val_loss < best_val - 1e-4:
                best_val, bad_epochs = val_loss, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if not k.startswith("encoder.")
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # -- calibration on val -------------------------------------------------
    model.eval()
    metrics: dict[str, Any] = {"history": history, "train_seconds": time.time() - t0}
    if val_rows:
        with torch.no_grad():
            probs, tgts, qids = _collect(model, val_rows)
        metrics["val_ece_before"] = _ece(probs, tgts)

        model.temperature = fit_temperature(model, val_rows)
        # temperature scales logits, not probs — recompute the distributions
        logits = model.forward_logits(
            [r.state for r in val_rows], [r.questions for r in val_rows]
        )
        probs_t = np.zeros_like(probs)
        for i, pair in enumerate(logits):
            p = torch.softmax(
                pair["logits"] / model.temperature, dim=-1
            ).detach().cpu().numpy()
            probs_t[i, : len(p)] = p
        metrics["val_ece_temperature"] = _ece(probs_t, tgts)
        metrics["temperature"] = model.temperature

        iso = fit_isotonic(probs, tgts, qids)
        probs_i = apply_isotonic(probs, qids, iso)
        metrics["val_ece_isotonic"] = _ece(probs_i, tgts)
        model.calibration = {
            "method": cfg.get("calibration", "temperature"),
            "temperature": model.temperature,
            "isotonic": iso,
        }
    else:
        metrics["val_ece_before"] = None

    ckpt = model.save(out_dir / "checkpoint")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    metrics.update(
        {
            "checkpoint": str(ckpt),
            "metrics_path": str(metrics_path),
            "device": str(device),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "steps": step,
        }
    )

    # -- MLflow (best effort) ------------------------------------------------
    def _log() -> None:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="jev-clf-train"):
            mlflow.log_params(
                {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(metrics_path))

    _mlflow_log(_log)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the jev_clf OptionScorer")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "jev_clf.yaml"))
    ap.add_argument("--out", default=str(ARTIFACTS / "run"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rows: list[DecisionRow] = []
    for path in cfg.get("data", {}).get("train_files", []):
        p = REPO_ROOT / path
        if p.exists():
            rows.extend(read_rows(p))
        else:
            print(f"[data] missing, skipped: {p}")
    metrics = train(rows, cfg, args.out)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
