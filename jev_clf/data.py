"""Ground-truth loaders: real fact-verification datasets -> DecisionRow.

Each loader produces rows with ``label_source="ground_truth"`` and a state of
``{"claim": ..., "evidence": [{"title": ..., "text": ...}, ...]}``. Raw data is
cached under ``data/factcheck/raw/`` (HF datasets cache + the SciFact tarball);
a second call never re-downloads.

Sources and label mappings (all onto FACTCHECK_LABELS):

- ``fever``        -> ``pietrolesci/nli_fever``. FEVER pairs already joined to
  their evidence sentence: ``premise`` is the claim, ``hypothesis`` the
  Wikipedia evidence sentence, ``fever_gold_label`` in
  {SUPPORTS, REFUTES, NOT ENOUGH INFO} maps 1:1. ``cid`` (claim id) is the
  group key so every evidence row of one claim stays in one split.
- ``vitaminc``     -> ``tals/vitaminc``. Claim + evidence sentence + ``page``
  title inline; ``label`` in {SUPPORTS, REFUTES, NOT ENOUGH INFO} maps 1:1.
  ``case_id`` groups the revisions of one base claim.
- ``scifact``      -> ``allenai/scifact`` release tarball (the HF repo is a
  legacy loading script that datasets>=4 refuses; the tarball ships
  ``claims_*.jsonl`` + ``corpus.jsonl`` together, so joining claims to their
  cited abstracts is a local join, not a retrieval pipeline). Annotated
  rationales: SUPPORT -> supported, CONTRADICT/REFUTE -> refuted; one row per
  (claim, evidence doc) using the annotated rationale sentences. Claims with
  no annotated evidence become ``not_enough_info`` rows whose evidence is the
  cited abstracts the annotators saw (empty when none are cited).
- ``climate_fever`` -> ``tdiggelm/climate_fever`` (the canonical
  ``climate_fever`` repo is a legacy script too). Claim + retrieved evidence
  sentences inline. {SUPPORTS, REFUTES, NOT_ENOUGH_INFO} map 1:1; DISPUTED
  claims are dropped — annotators disagreed on the verdict, so there is no
  ground truth to teach.

Skipped candidates: raw ``fever`` (claims + ``wiki_pages``) needs a Wikipedia
retrieval pass — out of scope; ``allenai/scifact`` and ``climate_fever`` via
``load_dataset`` are script-based and unsupported by datasets 5.x (SciFact is
loaded from its release tarball instead).

Regenerate:  uv run python -m jeff.data
"""
from __future__ import annotations

import json
import os
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "factcheck" / "raw"
OUT_PATH = REPO_ROOT / "data" / "factcheck" / "ground_truth.jsonl"

os.environ.setdefault("HF_DATASETS_CACHE", str(RAW_DIR))

from datasets import load_dataset  # noqa: E402  (after env var)

from .schema import (  # noqa: E402
    FACTCHECK_LABELS,
    FACTCHECK_QUESTION_ID,
    DecisionRow,
    label_space,
    make_factcheck_choice,
    one_hot,
    read_rows,
    write_rows,
)

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_DIR = RAW_DIR / "scifact"

SPLIT_NAMES = ("train", "val", "test")

# Original dataset label -> FACTCHECK_LABELS entry. See module docstring.
FEVER_LABEL_MAP = {
    "SUPPORTS": "supported",
    "REFUTES": "refuted",
    "NOT ENOUGH INFO": "not_enough_info",
}
VITAMINC_LABEL_MAP = dict(FEVER_LABEL_MAP)
SCIFACT_LABEL_MAP = {
    "SUPPORT": "supported",
    "CONTRADICT": "refuted",
    "REFUTE": "refuted",
}
CLIMATE_FEVER_LABEL_MAP = {
    "SUPPORTS": "supported",
    "REFUTES": "refuted",
    "NOT_ENOUGH_INFO": "not_enough_info",
}


def _norm_claim(text: str) -> str:
    """Canonical claim/evidence identity: lowercase, then collapse every run
    of non-word characters to one space. Strips case, punctuation, bracketed
    markup spans (``[NASA]`` -> ``nasa``) and ``[...]`` elision markers —
    Climate-FEVER annotator markup, not claim content."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def _row(
    *,
    source: str,
    row_id: str,
    group_id: str,
    claim: str,
    evidence: list[dict],
    label: str,
    meta: dict,
) -> DecisionRow:
    """Build one ground-truth row. ``split`` is a placeholder; assign_splits
    rewrites it."""
    return DecisionRow(
        row_id=row_id,
        source=source,
        split="train",
        group_id=group_id,
        state={"claim": claim, "evidence": evidence},
        questions={FACTCHECK_QUESTION_ID: make_factcheck_choice()},
        labels={
            FACTCHECK_QUESTION_ID: one_hot(list(FACTCHECK_LABELS), label)
        },
        label_source="ground_truth",
        meta=meta,
    )


def _cap_groups(rows: list[DecisionRow], max_rows: int, seed: int) -> list[DecisionRow]:
    """Deterministically subsample to <= max_rows without splitting a group."""
    if len(rows) <= max_rows:
        return rows
    groups: dict[str, list[DecisionRow]] = defaultdict(list)
    for r in rows:
        groups[r.group_id].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out: list[DecisionRow] = []
    for k in keys:
        if len(out) + len(groups[k]) > max_rows and out:
            continue
        out.extend(groups[k])
        if len(out) >= max_rows:
            break
    return out[:max_rows]

def _hf_revision(dataset_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id).sha
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------


def _load_fever(max_rows: int, seed: int) -> list[DecisionRow]:
    ds = load_dataset("pietrolesci/nli_fever")
    rev = _hf_revision("pietrolesci/nli_fever")
    rows: list[DecisionRow] = []
    for hf_split in ("train", "dev", "test"):
        if hf_split not in ds:
            continue
        for ex in ds[hf_split]:
            label = FEVER_LABEL_MAP.get(ex["fever_gold_label"])
            if label is None:
                continue
            rows.append(
                _row(
                    source="fever",
                    row_id=f"fever-{hf_split}-{ex['fid']}",
                    group_id=f"fever-claim-{ex['cid']}",
                    claim=ex["premise"],
                    evidence=[{"title": str(ex["fid"]), "text": ex["hypothesis"]}],
                    label=label,
                    meta={
                        "dataset": "pietrolesci/nli_fever",
                        "revision": rev,
                        "hf_split": hf_split,
                        "original_label": ex["fever_gold_label"],
                        "original_id": ex["fid"],
                        "claim_id": ex["cid"],
                    },
                )
            )
    return _cap_groups(rows, max_rows, seed)


def _load_vitaminc(max_rows: int, seed: int) -> list[DecisionRow]:
    ds = load_dataset("tals/vitaminc")
    rev = _hf_revision("tals/vitaminc")
    rows: list[DecisionRow] = []
    for hf_split in ("train", "validation", "test"):
        if hf_split not in ds:
            continue
        for ex in ds[hf_split]:
            label = VITAMINC_LABEL_MAP.get(ex["label"])
            if label is None:
                continue
            rows.append(
                _row(
                    source="vitaminc",
                    row_id=f"vitaminc-{ex['unique_id']}",
                    group_id=f"vitaminc-case-{ex['case_id']}",
                    claim=ex["claim"],
                    evidence=[{"title": ex["page"], "text": ex["evidence"]}],
                    label=label,
                    meta={
                        "dataset": "tals/vitaminc",
                        "revision": rev,
                        "hf_split": hf_split,
                        "original_label": ex["label"],
                        "original_id": ex["unique_id"],
                        "case_id": ex["case_id"],
                        "revision_type": ex["revision_type"],
                    },
                )
            )
    return _cap_groups(rows, max_rows, seed)


def _scifact_data_dir() -> Path:
    """Fetch + extract the SciFact release tarball once; reuse after."""
    data_dir = SCIFACT_DIR / "data"
    if not (data_dir / "corpus.jsonl").exists():
        SCIFACT_DIR.mkdir(parents=True, exist_ok=True)
        tar_path = SCIFACT_DIR / "data.tar.gz"
        if not tar_path.exists():
            urllib.request.urlretrieve(SCIFACT_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(SCIFACT_DIR, filter="data")
    return data_dir


def _load_scifact(max_rows: int, seed: int) -> list[DecisionRow]:
    data_dir = _scifact_data_dir()
    corpus: dict[int, dict] = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc

    rows: list[DecisionRow] = []
    for hf_split in ("train", "dev", "test"):
        path = data_dir / f"claims_{hf_split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                cid = ex["id"]
                group_id = f"scifact-claim-{cid}"
                if ex.get("evidence"):
                    for doc_id_str, entries in ex["evidence"].items():
                        doc = corpus.get(int(doc_id_str))
                        if doc is None:
                            continue
                        labels = {e["label"] for e in entries}
                        if len(labels) != 1:
                            continue  # conflicting rationales: no ground truth
                        orig = entries[0]["label"]
                        label = SCIFACT_LABEL_MAP.get(orig)
                        if label is None:
                            continue
                        sent_idx = sorted(
                            {i for e in entries for i in e["sentences"]}
                        )
                        text = " ".join(
                            doc["abstract"][i]
                            for i in sent_idx
                            if i < len(doc["abstract"])
                        )
                        rows.append(
                            _row(
                                source="scifact",
                                row_id=f"scifact-{hf_split}-{cid}-{doc_id_str}",
                                group_id=group_id,
                                claim=ex["claim"],
                                evidence=[{"title": doc["title"], "text": text}],
                                label=label,
                                meta={
                                    "dataset": "allenai/scifact",
                                    "revision": "release/latest",
                                    "hf_split": hf_split,
                                    "original_label": orig,
                                    "original_id": cid,
                                    "evidence_doc_id": int(doc_id_str),
                                },
                            )
                        )
                else:
                    evidence = [
                        {"title": corpus[d]["title"], "text": " ".join(corpus[d]["abstract"])}
                        for d in ex.get("cited_doc_ids", [])
                        if d in corpus
                    ]
                    rows.append(
                        _row(
                            source="scifact",
                            row_id=f"scifact-{hf_split}-{cid}-noinfo",
                            group_id=group_id,
                            claim=ex["claim"],
                            evidence=evidence,
                            label="not_enough_info",
                            meta={
                                "dataset": "allenai/scifact",
                                "revision": "release/latest",
                                "hf_split": hf_split,
                                "original_label": "NOINFO",
                                "original_id": cid,
                            },
                        )
                    )
    return _cap_groups(rows, max_rows, seed)


def _load_climate_fever(max_rows: int, seed: int) -> list[DecisionRow]:
    ds = load_dataset("tdiggelm/climate_fever")
    rev = _hf_revision("tdiggelm/climate_fever")
    names = ds["test"].features["claim_label"].names
    rows: list[DecisionRow] = []
    for hf_split in ds:
        for ex in ds[hf_split]:
            orig = names[ex["claim_label"]]
            label = CLIMATE_FEVER_LABEL_MAP.get(orig)
            if label is None:
                continue  # DISPUTED: no agreed ground truth
            evidence = [
                {"title": e["article"], "text": e["evidence"]}
                for e in ex["evidences"]
            ]
            rows.append(
                _row(
                    source="climate_fever",
                    row_id=f"climate_fever-{hf_split}-{ex['claim_id']}",
                    group_id=f"climate_fever-claim-{ex['claim_id']}",
                    claim=ex["claim"],
                    evidence=evidence,
                    label=label,
                    meta={
                        "dataset": "tdiggelm/climate_fever",
                        "revision": rev,
                        "hf_split": hf_split,
                        "original_label": orig,
                        "original_id": ex["claim_id"],
                    },
                )
            )
    return _cap_groups(rows, max_rows, seed)


LOADERS = {
    "fever": _load_fever,
    "vitaminc": _load_vitaminc,
    "scifact": _load_scifact,
    "climate_fever": _load_climate_fever,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dedupe_key(row: DecisionRow) -> tuple:
    """Normalized (claim, evidence-set) identity of a row."""
    claim = row.state.get("claim") if isinstance(row.state, dict) else None
    evidence = row.state.get("evidence") if isinstance(row.state, dict) else None
    ev_key = frozenset(
        _norm_claim(e.get("text", "")) for e in (evidence or [])
    )
    return (_norm_claim(claim or ""), ev_key)


def _dedupe(rows: list[DecisionRow]) -> tuple[list[DecisionRow], int]:
    """Drop exact (claim, evidence-set) duplicates, keeping the first
    occurrence. The survivor records ``meta["dup_count"]`` (itself included)
    and ``meta["dropped_row_ids"]``. Deterministic: input order is fixed."""
    seen: dict[tuple, DecisionRow] = {}
    out: list[DecisionRow] = []
    dropped = 0
    for r in rows:
        key = _dedupe_key(r)
        if key in seen:
            survivor = seen[key]
            survivor.meta["dup_count"] = survivor.meta.get("dup_count", 1) + 1
            survivor.meta.setdefault("dropped_row_ids", []).append(r.row_id)
            dropped += 1
        else:
            seen[key] = r
            out.append(r)
    return out, dropped


def load_ground_truth(
    sources: Iterable[str] = ("fever", "vitaminc", "scifact"),
    max_per_source: int = 2000,
    seed: int = 42,
) -> list[DecisionRow]:
    """Load real fact-verification rows, capped per source, deduped on
    (claim, evidence-set), split by group."""
    rows: list[DecisionRow] = []
    for src in sources:
        if src not in LOADERS:
            raise ValueError(f"unknown source {src!r}; have {sorted(LOADERS)}")
        rows.extend(LOADERS[src](max_per_source, seed))
    rows, dropped = _dedupe(rows)
    if dropped:
        print(f"dedupe: dropped {dropped} duplicate (claim, evidence) rows")
    return assign_splits(rows, seed=seed)


def _claim_components(rows: list[DecisionRow]) -> dict[str, str]:
    """Union-find over group_id: merge groups that share a normalized claim
    text, so identical claims can never straddle splits. Returns
    group_id -> component root."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    seen_claim: dict[str, str] = {}
    for r in rows:
        claim = r.state.get("claim") if isinstance(r.state, dict) else None
        if claim is None:
            continue
        key = _norm_claim(claim)
        if key in seen_claim:
            union(seen_claim[key], r.group_id)
        else:
            seen_claim[key] = r.group_id
    return {g: find(g) for g in {r.group_id for r in rows}}


def assign_splits(
    rows: list[DecisionRow],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> list[DecisionRow]:
    """Deterministically assign train/val/test by group. A group (and any
    group sharing its claim text) never straddles splits."""
    component = _claim_components(rows)
    comp_rows: dict[str, list[DecisionRow]] = defaultdict(list)
    for r in rows:
        comp_rows[component[r.group_id]].append(r)

    comps = sorted(comp_rows)
    random.Random(seed).shuffle(comps)

    total = len(rows)
    target = [r * total for r in ratios]
    counts = [0, 0, 0]
    assignment: dict[str, str] = {}
    for comp in comps:
        n = len(comp_rows[comp])
        # assign to the split with the largest remaining deficit
        deficits = [target[i] - counts[i] for i in range(3)]
        idx = max(range(3), key=lambda i: deficits[i])
        assignment[comp] = SPLIT_NAMES[idx]
        counts[idx] += n

    for r in rows:
        r.split = assignment[component[r.group_id]]
    return rows


# ---------------------------------------------------------------------------
# CLI: build data/factcheck/ground_truth.jsonl
# ---------------------------------------------------------------------------


def _main() -> None:
    sources = ("fever", "vitaminc", "scifact", "climate_fever")
    rows = load_ground_truth(sources=sources, max_per_source=500, seed=42)
    write_rows(OUT_PATH, rows)
    print(f"wrote {len(rows)} rows -> {OUT_PATH}")

    # per-source x per-label table
    table: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        label = max(r.labels[FACTCHECK_QUESTION_ID].items(), key=lambda kv: kv[1])[0]
        table[r.source][label] += 1
    header = ("source",) + FACTCHECK_LABELS + ("total",)
    print(("{:<14}" + "{:>16}" * 4).format(*header))
    for src in sources:
        c = table[src]
        print(
            ("{:<14}" + "{:>16}" * 4).format(
                src, *(c[l] for l in FACTCHECK_LABELS), sum(c.values())
            )
        )
    split_counts = Counter(r.split for r in rows)
    print("splits:", dict(split_counts))

    # acceptance checks
    loaded = read_rows(OUT_PATH)
    assert len(loaded) == len(rows), "round-trip row count mismatch"
    for r in loaded:
        assert label_space(r.questions[FACTCHECK_QUESTION_ID]) == list(
            FACTCHECK_LABELS
        ), f"{r.row_id}: bad label space"
    split_by_group: dict[str, set] = defaultdict(set)
    for r in loaded:
        split_by_group[r.group_id].add(r.split)
    straddlers = {g: s for g, s in split_by_group.items() if len(s) > 1}
    assert not straddlers, f"groups straddling splits: {straddlers}"

    claims_by_split: dict[str, set] = defaultdict(set)
    for r in loaded:
        if isinstance(r.state, dict) and "claim" in r.state:
            claims_by_split[r.split].add(_norm_claim(r.state["claim"]))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = claims_by_split[a] & claims_by_split[b]
        assert not overlap, f"{len(overlap)} claims appear in both {a} and {b}"
    print("all acceptance checks passed")


if __name__ == "__main__":
    _main()
