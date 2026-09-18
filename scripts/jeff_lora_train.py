"""LoRA fine-tune an instruct LM as the jeff fact-checking classifier.

The classifier is the model's own next-token distribution over the label
tokens — no head. This script teaches the model to put its mass there by
supervising ONLY the assistant turn (" <label>" + <|im_end|>) of each SFT
row; every prompt and padding position is masked to -100.

Prompt contract (aligned with scripts/jeff_lm_eval.py):
  * prompt_format: eval  — renders each row exactly as the eval harness:
    eval SYSTEM prompt + user text + "\\n\\nVerdict:" under the chat
    template with add_generation_prompt=True.
  * prompt_format: sft   — the row's own system/user messages verbatim.
  * completion_leading_space (default on) — the assistant content is
    " " + label, because the eval readout scores the first token of
    " "+label (label_variants prefers the mid-sentence form).

Usage (from the repo root):

    uv run python -m scripts.jeff_lora_train --config configs/jeff_lora.yaml
    uv run python -m scripts.jeff_lora_train --config configs/jeff_lora.yaml \
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
from scripts.jeff_lm_eval import SYSTEM as EVAL_SYSTEM  # noqa: E402

MLFLOW_URI = "http://127.0.0.1:5001"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sft_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = [r["row_id"] for r in rows if r.get("split") == "test"]
    assert not bad, f"test-split rows must never reach training: {bad[:3]}"
    return rows


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
    return {"input_ids": input_ids, "labels": labels, "weight": float(row.get("weight", 1.0)),
            "label_source": row.get("label_source", "")}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
        "weights": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
    }


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Token-mean CE over supervised positions, optionally weighted per row."""
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
    return (per_row * w).sum()


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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["optim"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = cfg["optim"]
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
    total_steps = steps_per_epoch * opt["epochs"]
    warmup = max(1, int(total_steps * opt["warmup_ratio"]))
    sched = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"[train] epochs={opt['epochs']} steps/epoch={steps_per_epoch} "
          f"total={total_steps} warmup={warmup} eff_batch={opt['batch_size'] * opt['grad_accum']}")

    history = []
    rng = random.Random(seed)
    gstep = 0
    model.train()
    for epoch in range(opt["epochs"]):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        accum_loss, accum_n = 0.0, 0
        t_ep = time.perf_counter()
        for bi, start in enumerate(range(0, len(order), opt["batch_size"])):
            idx = order[start : start + opt["batch_size"]]
            b = collate([train_ex[i] for i in idx], pad_id)
            labels = b.pop("labels").to(device)
            weights = b.pop("weights").to(device)
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            loss = masked_lm_loss(logits, labels, weights) / opt["grad_accum"]
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
