# researchMax gap audit — decision results

Generated: `2026-09-19T13:36:17.083307+00:00`  
question: `verdict`  
n = 9730 rows  
labels: not_enough_info, refuted, supported

## Inputs

| file | sha256 | rows |
|---|---|---|
| gold: `/Users/djl/Desktop/jev_clf/data/factcheck/eval_large.jsonl` | `7e46efac6d6adf50…` | 9730 |
| ours_large: `/Users/djl/Desktop/jev_clf/data/factcheck/preds_ours_large.jsonl` | `36488db0effedbfc…` | 9730 |
| jev_large: `/Users/djl/Desktop/jev_clf/data/factcheck/preds_jev_large.jsonl` | `1d7b22a8d1c43afd…` | 9730 |

python 3.12.13 · numpy 2.5.3 · scipy 1.18.1

## Headline metrics (recomputed)

| arm | n | accuracy | macro F1 | NLL | Brier |
|---|---|---|---|---|---|
| ours_large | 9730 | 0.818294 (7962/9730) | 0.778875 | 0.553219 | 0.283924 |
| jev_large | 9730 | 0.828263 (8059/9730) | 0.799425 | 1.643788 | 0.274950 |

_ours_large: 71 exact argmax tie(s); accuracy under sorted-label tie-break would be 0.817986 (insertion-order policy used above)._

_jev_large: 12 exact argmax tie(s); accuracy under sorted-label tie-break would be 0.828366 (insertion-order policy used above)._

## Calibration (ECE, lower is better)

| arm | top-label 10-bin | stored-conf 10-bin | adaptive equal-count | classwise OvR macro |
|---|---|---|---|---|
| ours_large | 0.08075 | 0.08075 | 0.07907 | 0.05591 |
| jev_large | 0.09321 | 0.07896 | 0.09321 | 0.06457 |

Classwise one-vs-rest ECE:

| arm | not_enough_info | refuted | supported |
|---|---|---|---|
| ours_large | 0.05489 | 0.05953 | 0.05331 |
| jev_large | 0.07645 | 0.05390 | 0.06335 |

## Selective risk (stored confidence, ties fully included)

| arm | target coverage | threshold | realized coverage | n | risk |
|---|---|---|---|---|---|
| ours_large | 0.5 | 0.9765 | 0.5014 | 4879 | 0.0541 |
| ours_large | 0.8 | 0.8079 | 0.8000 | 7784 | 0.1138 |
| ours_large | 0.9 | 0.6391 | 0.9001 | 8758 | 0.1459 |
| jev_large | 0.5 | 0.9900 | 0.5207 | 5066 | 0.0568 |
| jev_large | 0.8 | 0.7900 | 0.8005 | 7789 | 0.1064 |
| jev_large | 0.9 | 0.5400 | 0.9022 | 8778 | 0.1355 |

## Recall by gold class

| arm | class | n | recall |
|---|---|---|---|
| ours_large | not_enough_info | 2106 | 0.5783 |
| ours_large | refuted | 2593 | 0.7941 |
| ours_large | supported | 5031 | 0.9312 |
| jev_large | not_enough_info | 2106 | 0.7118 |
| jev_large | refuted | 2593 | 0.8049 |
| jev_large | supported | 5031 | 0.8891 |

## Accuracy by source

| arm | source | n | accuracy |
|---|---|---|---|
| ours_large | climate_fever | 859 | 0.6694 |
| ours_large | fever | 3910 | 0.8422 |
| ours_large | scifact | 982 | 0.8870 |
| ours_large | vitaminc | 3979 | 0.8100 |
| jev_large | climate_fever | 859 | 0.5914 |
| jev_large | fever | 3910 | 0.8673 |
| jev_large | scifact | 982 | 0.8615 |
| jev_large | vitaminc | 3979 | 0.8329 |

## Accuracy by passage count

| arm | passages | n | accuracy |
|---|---|---|---|
| ours_large | 0 | 214 | 1.0000 |
| ours_large | 1 | 8643 | 0.8285 |
| ours_large | 2 | 11 | 0.9091 |
| ours_large | 3 | 2 | 1.0000 |
| ours_large | 4 | 1 | 0.0000 |
| ours_large | 5 | 859 | 0.6694 |
| jev_large | 0 | 214 | 1.0000 |
| jev_large | 1 | 8643 | 0.8477 |
| jev_large | 2 | 11 | 0.9091 |
| jev_large | 3 | 2 | 0.0000 |
| jev_large | 4 | 1 | 0.0000 |
| jev_large | 5 | 859 | 0.5914 |

## Paired comparison (same rows)

- A = `ours_large` acc 0.818294; B = `jev_large` acc 0.828263; Δ(B−A) = +0.009969
- McNemar exact: B-only 594, A-only 497, discordant 1091, p = 0.003637
- Paired bootstrap 95% CI on Δ: [+0.0033, +0.0165] (6000 resamples, seed 11)

## Confusion (gold × predicted)

### ours_large

| gold \ pred | not_enough_info | refuted | supported |
|---|---|---|---|
| not_enough_info | 1218 | 375 | 513 |
| refuted | 216 | 2059 | 318 |
| supported | 164 | 182 | 4685 |

### jev_large

| gold \ pred | not_enough_info | refuted | supported |
|---|---|---|---|
| not_enough_info | 1499 | 306 | 301 |
| refuted | 345 | 2087 | 161 |
| supported | 406 | 152 | 4473 |


## Historical published summaries (verbatim, NOT recomputed)

verbatim copies of previously published summaries; NOT recomputed here and kept separate from the audited numbers above

### `/Users/djl/Desktop/jev_clf/results/jev_large.json`

```json
{
  "accuracy": 0.8282631038026721,
  "brier": 0.274950058136358,
  "confusion": {
    "not_enough_info": {
      "not_enough_info": 1499,
      "refuted": 306,
      "supported": 301
    },
    "refuted": {
      "not_enough_info": 345,
      "refuted": 2087,
      "supported": 161
    },
    "supported": {
      "not_enough_info": 406,
      "refuted": 152,
      "supported": 4473
    }
  },
  "ece": 0.07895580678314494,
  "labels": [
    "not_enough_info",
    "refuted",
    "supported"
  ],
  "macro_f1": 0.7994254905106022,
  "n": 9730,
  "per_label": {
    "not_enough_info": {
      "f1": 0.6882460973370064,
      "precision": 0.6662222222222223,
      "recall": 0.7117758784425451,
      "support": 2106
    },
    "refuted": {
      "f1": 0.8123783573374854,
      "precision": 0.8200392927308447,
      "recall": 0.8048592364057077,
      "support": 2593
    },
    "supported": {
      "f1": 0.8976520168573149,
      "precision": 0.9063829787234042,
      "recall": 0.889087656529517,
      "support": 5031
    }
  }
}
```


## Formula definitions

- **key**: (row_id, question_id); pred keys for the audited question must equal gold keys exactly; duplicates in either file abort
- **labels**: sorted union taken from the first gold row's labels[question]; every gold/pred probability map must match it exactly
- **gold**: labels[qid] must be finite, nonnegative, sum to 1 within 1e-06, and be exactly one-hot
- **probs**: finite, nonnegative, sum to 1 within 1e-06; top-1 = max(probs.items(), key=kv[1]) so exact ties resolve to the FIRST INSERTED key (existing eval policy); n_exact_argmax_ties and accuracy_alt_sorted_tiebreak report tie prevalence and the sorted-order counterfactual
- **confidence**: stored confidence required, finite, within [0,1] +/- 1e-09, clipped for binning
- **accuracy**: mean(argmax(probs) == gold_label)
- **macro_f1**: mean over classes of 2PR/(P+R); empty class contributes 0
- **nll**: mean(-log(max(p_gold, 1e-15)))
- **brier_multiclass**: mean over rows of sum_c (p_c - y_c)^2
- **ece_top_label_10bin**: 10 equal-width bins on [0,1] over max(probs); sum_b (n_b/N) * |acc_b - conf_b|
- **ece_stored_confidence_10bin**: same binning over the stored 'confidence' field, reported separately
- **ece_classwise_ovr_10bin**: per class c, 10-bin ECE of p_c vs 1[gold==c]; macro = mean over classes
- **ece_adaptive_equal_count_10bin**: 10 equal-count bins over ascending stable rank of max(probs); ties keep input order and may straddle a boundary
- **selective_risk**: for target c: k=ceil(c*N), tau=kth-largest stored confidence, keep ALL rows with conf>=tau (tie group fully included, so realized coverage >= c); risk = 1 - accuracy on kept rows
- **mcnemar**: exact two-sided binomtest on discordant pairs (b_only_correct, a_only_correct) on identical rows
- **bootstrap**: 6000 paired row resamples, seed 11, percentile 95% CI on acc(B)-acc(A)
