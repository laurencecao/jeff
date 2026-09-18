"""LoRA fine-tune an instruct LM as the jeff fact-checking classifier.

The classifier is the model's own next-token distribution over the label
tokens — no head. This script teaches the model to put its mass there by
supervising ONLY the assistant turn (" <label>" + <|im_end|>) of each SFT
row; every prompt and padding position is masked to -100.

Prompt contract (aligned with scripts/jev_clf_lm_eval.py):
  * prompt_format: eval  — renders each row exactly as the eval harness:
    eval SYSTEM prompt + user text + "\\n\\nVerdict:" under the chat
    template with add_generation_prompt=True.
  * prompt_format: sft   — the row's own system/user messages verbatim.
  * completion_leading_space (default on) — the assistant content is
    " " + label, because the eval readout scores the first token of
    " "+label (label_variants prefers the mid-sentence form).

Usage (from the repo root):

    uv run python -m scripts.jev_clf_lora_train --config configs/jev_clf_lora.yaml
    uv run python -m scripts.jev_clf_lora_train --config configs/jev_clf_lora.yaml \
        --max-rows 64 --epochs 1 --out-dir /tmp/lora_smoke   # smoke test

Never trains or evaluates on the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The eval SYSTEM prompt is imported, not copied, so the training prompt can
# never drift from the harness that scores the adapter.
from scripts.jev_clf_lm_eval import SYSTEM as EVAL_SYSTEM  # noqa: E402

MLFLOW_URI = "http://127.0.0.1:5001"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sft_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = [r["row_id"] for r in rows if r.get("split") == "test"]
    assert not bad, f"test-split rows must never reach training: {bad[:3]}"
    return rows


def label_first_token_ids(tok, labels: list[str], leading_space: bool) -> list[int] | None:
    """First-token id per label, or None if the label space is not separable.

    Mirrors jev_clf/readout.py: first-token readout is only meaningful when
    every label has a DISTINCT first token. Score labels (" 0".." 3") all
    tokenize to [220, 15/16/17/18], sharing token 220, so a first-token
    soft target over them would be degenerate. Returning None there keeps the
    soft term off rather than silently training on a collapsed distribution.

    `leading_space` must match the completion convention: the assistant turn
    is " " + label, and the readout scores that same first token.
    """
    ids = []
    for lab in labels:
        toks = tok((" " if leading_space else "") + lab, add_special_tokens=False)["input_ids"]
        if not toks:
            return None
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        return None
    return ids


def build_example(tok, row: dict, cfg: dict) -> dict:
    """Tokenize one SFT row into input_ids/labels with prompt masked to -100."""
    msgs = row["messages"]
    sys_msg = next(m for m in msgs if m["role"] == "system")
    user_msg = next(m for m in msgs if m["role"] == "user")
    asst_msg = next(m for m in msgs if m["role"] == "assistant")
    assert asst_msg["content"].strip() == row["label"], (
        f"{row['row_id']}: assistant content {asst_msg['content']!r} != label {row['label']!r}"
    )

    if cfg["data"]["prompt_format"] == "eval":
        system = EVAL_SYSTEM
        user = user_msg["content"] + "\n\nVerdict:"
    else:  # "sft": the row's own messages verbatim
        system = sys_msg["content"]
        user = user_msg["content"]

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = (" " if cfg["data"]["completion_leading_space"] else "") + row["label"]

    # Tokenize prompt and completion separately so the boundary is exact —
    # BPE could otherwise merge the trailing "\n" of the generation prompt
    # with the leading space of the completion.
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        comp_ids = comp_ids + [tok.eos_token_id]

    max_len = cfg["max_length"]
    if len(prompt_ids) + len(comp_ids) > max_len:
        # Right-truncate the prompt exactly like the eval harness does
        # (tok(..., truncation=True) drops tail tokens), keeping the
        # completion intact and adjacent to whatever context survives.
        prompt_ids = prompt_ids[: max_len - len(comp_ids)]

    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids

    # --- soft targets -----------------------------------------------------
    # The distill rows carry the teacher's FULL distribution, e.g.
    # {"refuted": 0.93, "not_enough_info": 0.07, "supported": 0.0}. Training on
    # the argmax alone discards exactly the hedging that separates a confident
    # refutation from "topically relevant but not established". Soft targets
    # keep that signal; the class index is the first supervised position.
    # -1 marks "no usable soft target" so the soft term is simply skipped.
    soft: list[float] = []
    soft_ids: list[int] = []
    soft_pos = -1
    probs = row.get("target_probs") or {}
    label_space = row.get("label_space") or []
    if probs and label_space:
        ft = label_first_token_ids(tok, list(label_space), bool(cfg["data"]["completion_leading_space"]))
        if ft is not None:
            vals = [float(probs.get(lab, 0.0)) for lab in label_space]
            total = sum(vals)
            if total > 0:
                soft = [v / total for v in vals]
                soft_ids = ft
                soft_pos = len(prompt_ids)

    return {"input_ids": input_ids, "labels": labels, "weight": float(row.get("weight", 1.0)),
            "label_source": row.get("label_source", ""),
            "soft": soft, "soft_ids": soft_ids, "soft_pos": soft_pos}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)

    # Soft targets: one padded row per example, position 0 when absent so the
    # gather is always in range (the mask keeps those rows out of the loss).
    n_lab = max((len(b["soft"]) for b in batch), default=0)
    soft, soft_ids, soft_pos, soft_mask = [], [], [], []
    for b in batch:
        s = b["soft"]
        if s and n_lab:
            soft.append(s + [0.0] * (n_lab - len(s)))
            soft_ids.append(list(b["soft_ids"]) + [0] * (n_lab - len(b["soft_ids"])))
            soft_pos.append(b["soft_pos"])
            soft_mask.append(1.0)
        else:
            soft.append([0.0] * n_lab)
            soft_ids.append([0] * n_lab)
            soft_pos.append(0)
            soft_mask.append(0.0)

    out = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
        "weights": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        "soft_mask": torch.tensor(soft_mask, dtype=torch.float32),
        "soft_pos": torch.tensor(soft_pos, dtype=torch.long),
    }
    if n_lab:
        out["soft"] = torch.tensor(soft, dtype=torch.float32)
        out["soft_ids"] = torch.tensor(soft_ids, dtype=torch.long)
    return out


def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    soft: torch.Tensor | None = None,
    soft_ids: torch.Tensor | None = None,
    soft_pos: torch.Tensor | None = None,
    soft_mask: torch.Tensor | None = None,
    soft_weight: float = 0.0,
) -> torch.Tensor:
    """Token-mean CE over supervised positions, plus optional soft-target KL.

    The CE term supervises the assistant turn exactly as before, so an
    unmodified config trains bit-identically to the previous script.

    When `soft` carries the teacher's distribution over the label space and
    `soft_weight` > 0, a KL(teacher || student) term is added at the first
    supervised position, reading the student's mass off the FIRST TOKEN of each
    label. The teacher distribution is the soft-target signal produced by
    distillation; without this term it is discarded and only the argmax is
    learned -- which is precisely the "confident but wrong at the boundary"
    failure this arm is meant to fix.
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    tok_loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.clamp(min=0).reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)
    per_row = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
    w = weights / weights.sum().clamp(min=1e-8)
    loss = (per_row * w).sum()

    if (
        soft is not None
        and soft_ids is not None
        and soft_pos is not None
        and soft_mask is not None
        and soft_weight > 0
        and soft_mask.sum() > 0
    ):
        # Position p is predicted by logits index p-1.
        rows = soft_mask > 0
        idx = rows.nonzero(as_tuple=True)[0]
        pos = (soft_pos[idx] - 1).clamp(min=0)
        # Index straight down to [n_rows, n_labels]. Do NOT slice
        # logits[idx].float() -- that materialises a full [batch, seq, vocab]
        # copy and upcasts it to fp32 (~8 GiB here), which OOMs the MPS pool.
        # Advanced indexing over (row, position, label-token) builds only the
        # handful of logits this term actually reads.
        lab_logits = logits[
            idx.view(-1, 1), pos.view(-1, 1), soft_ids[idx].clamp(min=0)
        ].float()
        logp = torch.log_softmax(lab_logits, dim=-1)
        teacher = soft[idx]
        # KL(teacher || student). Exact teacher zeros contribute nothing, but
        # 0 * log(0) must not become NaN, hence the clamp before the log.
        kl = (teacher * (torch.log(teacher.clamp_min(1e-12)) - logp)).sum(1)
        # Weighted MEAN over the rows that have a soft target (not a sum over a
        # fraction of the batch), so soft_target_weight is directly comparable
        # to the CE term: 1.0 means the KL counts as much as the CE does.
        rw = weights[idx]
        loss = loss + soft_weight * (kl * rw).sum() / rw.sum().clamp(min=1e-8)
    return loss


def evaluate_loss(model, examples: list[dict], pad_id: int, batch_size: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            b = collate(examples[i : i + batch_size], pad_id)
            labels = b.pop("labels").to(device)
            weights = b.pop("weights").to(device)
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            total += float(masked_lm_loss(logits, labels, weights)) * len(examples[i : i + batch_size])
            count += len(examples[i : i + batch_size])
    model.train()
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="subsample train rows (smoke test)")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--out-dir", default=None, help="override config out_dir")
    ap.add_argument("--resume", action="store_true",
                    help="skip epochs already completed in out_dir/train_state.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["optim"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = cfg["optim"]
    # 0.0 keeps the loss exactly the hard-label CE it has always been, so any
    # config without this key reproduces the previous training bit-for-bit.
    soft_weight = float(opt.get("soft_target_weight", 0.0))
    seed = opt["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float32
    t_start = time.perf_counter()

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    train_rows = load_sft_rows(ROOT / cfg["data"]["train"])
    val_rows = load_sft_rows(ROOT / cfg["data"]["val"])
    if args.max_rows:
        train_rows = train_rows[: args.max_rows]
    print(f"[data] train={len(train_rows)} val={len(val_rows)} "
          f"prompt_format={cfg['data']['prompt_format']} "
          f"leading_space={cfg['data']['completion_leading_space']}")

    src_w = cfg["data"].get("label_source_weights") or {}
    train_ex = []
    for r in train_rows:
        ex = build_example(tok, r, cfg)
        ex["weight"] *= float(src_w.get(ex["label_source"], 1.0))
        train_ex.append(ex)
    val_ex = [build_example(tok, r, cfg) for r in val_rows]
    n_soft = sum(1 for e in train_ex if e["soft"])
    print(f"[data] soft targets usable on {n_soft}/{len(train_ex)} train rows "
          f"(soft_target_weight={soft_weight})")
    lens = np.array([len(e["input_ids"]) for e in train_ex])
    print(f"[data] seq len mean={lens.mean():.0f} p95={np.percentile(lens, 95):.0f} "
          f"max={lens.max()} truncated={(lens >= cfg['max_length']).sum()}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=list(cfg["lora"]["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # --- resume from a previous epoch checkpoint --------------------------
    # Only valid because the LoRA is saved (not the optimizer state), so this
    # resumes from the last COMPLETED epoch with a fresh optimizer. Acceptable
    # here: the cosine schedule is short and the run is 2 epochs. It is not a
    # bit-exact continuation and must not be described as one.
    start_epoch = 0
    prior_history: list[dict] = []
    prior_val_loss = None
    state_path = out_dir / "train_state.json"
    if args.resume and state_path.exists() and (out_dir / "adapter_model.safetensors").exists():
        prior = json.loads(state_path.read_text())
        start_epoch = int(prior.get("epochs_done", 0))
        prior_history = list(prior.get("history", []))
        prior_val_loss = prior.get("val_loss")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, out_dir, is_trainable=True)
        print(f"[resume] loaded {out_dir} at epoch {start_epoch} "
              f"(prior val_loss={prior_val_loss}); training epochs "
              f"{start_epoch}..{opt['epochs'] - 1}", flush=True)
    if cfg.get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={n_train:,} total={n_total:,} frac={n_train / n_total:.4%}")

    # --- masking verification: the single most likely silent bug ----------
    b0 = collate(train_ex[: opt["batch_size"]], pad_id)
    n_sup = int((b0["labels"] != -100).sum())
    sup_tokens = b0["labels"][b0["labels"] != -100].tolist()
    print(f"[mask-check] batch0 supervised tokens={n_sup} "
          f"(of {b0['labels'].numel()} positions)")
    print(f"[mask-check] supervised text={tok.decode(sup_tokens)!r}")
    print(f"[mask-check] expected labels={[r['label'] for r in train_rows[: opt['batch_size']]]}")
    per_row_sup = (b0["labels"] != -100).sum(1).tolist()
    print(f"[mask-check] supervised tokens per row={per_row_sup}")
    assert all(1 <= s <= 8 for s in per_row_sup), "masking looks wrong: too many supervised tokens"
    # ----------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=opt["lr"], weight_decay=opt["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_ex) / (opt["batch_size"] * opt["grad_accum"]))
    remaining_epochs = opt["epochs"] - start_epoch
    total_steps = steps_per_epoch * remaining_epochs
    warmup = max(1, int(total_steps * opt["warmup_ratio"]))
    sched = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"[train] epochs={opt['epochs']} (starting at {start_epoch}) "
          f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup} "
          f"eff_batch={opt['batch_size'] * opt['grad_accum']}")

    history = list(prior_history)
    rng = random.Random(seed)
    gstep = 0
    model.train()
    for epoch in range(start_epoch, opt["epochs"]):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        accum_loss, accum_n = 0.0, 0
        t_ep = time.perf_counter()
        for bi, start in enumerate(range(0, len(order), opt["batch_size"])):
            idx = order[start : start + opt["batch_size"]]
            b = collate([train_ex[i] for i in idx], pad_id)
            labels = b.pop("labels").to(device)
            weights = b.pop("weights").to(device)
            soft = b.pop("soft", None)
            soft_ids = b.pop("soft_ids", None)
            soft_pos = b.pop("soft_pos", None)
            soft_mask = b.pop("soft_mask", None)
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            loss = masked_lm_loss(
                logits, labels, weights,
                soft=None if soft is None else soft.to(device),
                soft_ids=None if soft_ids is None else soft_ids.to(device),
                soft_pos=None if soft_pos is None else soft_pos.to(device),
                soft_mask=None if soft_mask is None else soft_mask.to(device),
                soft_weight=soft_weight,
            ) / opt["grad_accum"]
            loss.backward()
            accum_loss += float(loss) * opt["grad_accum"]
            accum_n += 1
            if accum_n == opt["grad_accum"] or start + opt["batch_size"] >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), opt["max_grad_norm"])
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 20 == 0 or gstep == 1:
                    avg = accum_loss / accum_n
                    lr_now = sched.get_last_lr()[0]
                    print(f"[train] epoch={epoch} step={gstep}/{total_steps} "
                          f"loss={avg:.4f} lr={lr_now:.2e} "
                          f"elapsed={time.perf_counter() - t_start:.0f}s", flush=True)
                    history.append({"step": gstep, "epoch": epoch, "loss": avg, "lr": lr_now})
                accum_loss, accum_n = 0.0, 0
        val_loss = evaluate_loss(model, val_ex, pad_id, opt["batch_size"], device)
        print(f"[train] epoch={epoch} done val_loss={val_loss:.4f} "
              f"epoch_time={time.perf_counter() - t_ep:.0f}s", flush=True)
        history.append({"step": gstep, "epoch": epoch, "val_loss": val_loss})

        # Checkpoint at every epoch boundary. Colab sessions are recycled and
        # /content is wiped with them, so a run that only saves at the very end
        # can lose hours of compute in a single recycle event. Writing here
        # means a kill mid-epoch costs at most one epoch, not the whole run.
        # --resume continues from the highest epoch already on disk.
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        (out_dir / "train_state.json").write_text(
            json.dumps({"epochs_done": epoch + 1, "gstep": gstep,
                        "val_loss": val_loss, "history": history}) + "\n"
        )
        print(f"[ckpt] saved epoch {epoch + 1}/{opt['epochs']} to {out_dir}", flush=True)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    wall = time.perf_counter() - t_start
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    metrics = {
        "model": cfg["model"],
        "adapter_dir": str(out_dir),
        "trainable_params": n_train,
        "total_params": n_total,
        "trainable_frac": n_train / n_total,
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "epochs": opt["epochs"],
        "effective_batch": opt["batch_size"] * opt["grad_accum"],
        "lr": opt["lr"],
        "seed": seed,
        "wall_time_s": wall,
        "peak_rss_gb": peak_rss_gb,
        "final_val_loss": next(h["val_loss"] for h in reversed(history) if "val_loss" in h),
        "history": history,
        "config": cfg,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[done] adapter -> {out_dir} wall={wall:.0f}s peak_rss={peak_rss_gb:.1f}GB")

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
            mlflow.log_params({
                "model": cfg["model"], "lora_r": cfg["lora"]["r"],
                "lora_alpha": cfg["lora"]["alpha"], "lora_dropout": cfg["lora"]["dropout"],
                "targets": ",".join(cfg["lora"]["target_modules"]),
                "lr": opt["lr"], "epochs": opt["epochs"],
                "eff_batch": opt["batch_size"] * opt["grad_accum"],
                "max_length": cfg["max_length"], "seed": seed,
                "prompt_format": cfg["data"]["prompt_format"],
                "leading_space": cfg["data"]["completion_leading_space"],
                "label_source_weights": json.dumps(src_w),
            })
            for h in history:
                if "loss" in h:
                    mlflow.log_metric("train_loss", h["loss"], step=h["step"])
                if "val_loss" in h:
                    mlflow.log_metric("val_loss", h["val_loss"], step=h["step"])
            mlflow.log_metrics({
                "trainable_frac": n_train / n_total,
                "wall_time_s": wall,
                "peak_rss_gb": peak_rss_gb,
            })
            mlflow.log_artifact(str(out_dir / "train_metrics.json"))
            mlflow.set_tag("arm", "lora_lm")
    except Exception as exc:  # best-effort; tracker may be down
        print(f"[mlflow] skipped: {exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    main()
