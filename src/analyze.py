"""
analyze.py
==========
Statistical analysis and logistic probing for GRAP gender-bias scores.

Reads JSONL score files and outputs per-model analysis + paper figures.

Per-model outputs (results/analysis/{model_id}/):
  stats.csv                              Wilcoxon p-values + Cohen's d
  univariate_auc.csv / .png             per-dimension univariate AUC
  delta_bar.png / mean_comparison.png   score visualisations
  logistic_weights_cherry10.csv         logistic coefficients (COMMON_DIMS)
  logistic_weights_cherry10_rm_outlier200.csv  after removing top-200 loss pairs

Paper figures (results/analysis/):
  delta_bar_std.png          score delta ± std, all 4 models, COMMON_DIMS
  coef_all_models.png        logistic coefficients, top male/female dims
  prob_kde_response_only.png P(female) KDE, response-only, all 20 dims
  metrics_table.png          AUC/Acc table, full vs response-only, all 20 dims
  roc_response_only.png      ROC curves, response-only, all 20 dims

Response-only combined ROC figures (results/analysis/):
  roc_response_only_all.png          2×2 per-model, COMMON_DIMS, original+cleaned
  roc_response_only_all_combined.png overlay all 4 models, original+cleaned
  roc_response_only_rm_outliers.png  overlay all 4 models, cleaned only

Usage:
  python src/analyze.py --score_dir results/scores/full   --out_dir results/analysis
  python src/analyze.py --score_dir results/scores/response_only --out_dir results/analysis
"""

import argparse
import csv
import glob
import json
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Common constants ───────────────────────────────────────────────────────────

# 10 dimensions significant (p<0.05 Wilcoxon) across all 4 evaluated models
COMMON_DIMS = [
    "communal_care_trait_attribution",
    "agentic_status_trait_attribution",
    "emotional_vulnerability",
    "warmth_respect",
    "need_for_protection_caution",
    "competence",
    "concern_risk_assessment",
    "negotiation_assertiveness",
    "agency_autonomy",
    "risk_tolerance",
]

# Fixed dim sets for the coef_all_models figure (paper Figure 2).
# Male dims ranked #1→#6 by decreasing average |coefficient|;
# Female dims ranked #1→#4. Order matches the finalized paper figure.
COEF_MALE_DIMS = [
    "agentic_status_trait_attribution",   # #1
    "agency_autonomy",                    # #2
    "capability_attribution",             # #3
    "self_confidence",                    # #4
    "competence",                         # #5
    "agency_empowerment",                 # #6
]
COEF_FEMALE_DIMS = [
    "communal_care_trait_attribution",    # #1
    "risk_tolerance",                     # #2
    "need_for_protection_caution",        # #3
    "warmth_respect",                     # #4
]

BASE_MODELS = [
    ("gpt-5.4-mini",           "GPT-5.4-mini"),
    ("claude-sonnet-4-5",      "Claude-Sonnet-4.5"),
    ("gemini-2.5-flash",       "Gemini-2.5-Flash"),
    ("llama-3.3-70b-instruct", "LLaMA-3.3-70B"),
]
RO_MODELS = [
    ("gpt-5.4-mini_response_only",           "GPT-5.4-mini"),
    ("claude-sonnet-4-5_response_only",      "Claude-Sonnet-4.5"),
    ("gemini-2.5-flash_response_only",       "Gemini-2.5-Flash"),
    ("llama-3.3-70b-instruct_response_only", "LLaMA-3.3-70B"),
]

PALETTE     = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]
LINESTYLES  = ["-", "--", ":", "-."]
C_MALE      = "#3DAA7A"
C_FEMALE    = "#9B59B6"
C_MAN_BAR   = "#4878D0"
C_WOMAN_BAR = "#EE854A"

LABEL_SHORT = {
    "communal_care_trait_attribution":    "Communal\nCare",
    "agentic_status_trait_attribution":   "Agentic\nStatus",
    "need_for_protection_caution":        "Need for\nProtection",
    "concern_risk_assessment":            "Concern Risk\nAssessment",
    "emotional_vulnerability":            "Emotional\nVulnerability",
    "negotiation_assertiveness":          "Negotiation\nAssertiveness",
    "agency_autonomy":                    "Agency\nAutonomy",
    "warmth_respect":                     "Warmth\nRespect",
    "risk_tolerance":                     "Risk\nTolerance",
    "competence":                         "Competence",
    "self_confidence":                    "Self\nConfidence",
    "capability_attribution":             "Capability\nAttribution",
    "agency_empowerment":                 "Agency\nEmpowerment",
    "leadership_potential":               "Leadership\nPotential",
    "professional_authority_credibility": "Professional\nAuthority",
}


def _slabel(key: str) -> str:
    return LABEL_SHORT.get(key, key.replace("_", " ").title())


def _label(key: str) -> str:
    return key.replace("_", " ").title()


def _sig_marker(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = rec.get("scores")
            if not isinstance(scores, dict):
                continue
            if all(isinstance(v, dict) and "score_a" in v and "score_b" in v
                   and not v.get("error") for v in scores.values()):
                records.append(rec)
    return records


def extract_scores(records: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    score_keys = list(records[0]["scores"].keys())
    man_rows, woman_rows = [], []
    for rec in records:
        man_rows.append([float(rec["scores"][k]["score_a"]) for k in score_keys])
        woman_rows.append([float(rec["scores"][k]["score_b"]) for k in score_keys])
    return score_keys, np.array(man_rows), np.array(woman_rows)


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(score_keys, man_scores, woman_scores) -> dict:
    out = {}
    for i, key in enumerate(score_keys):
        m = man_scores[:, i]
        w = woman_scores[:, i]
        delta = float(m.mean() - w.mean())
        d     = m - w
        p     = 1.0
        if not np.all(d == 0):
            try:
                _, p = stats.wilcoxon(m, w, alternative="two-sided")
            except ValueError:
                pass
        pooled = float(np.sqrt((m.std(ddof=1)**2 + w.std(ddof=1)**2) / 2))
        out[key] = {
            "mean_man":    float(m.mean()),
            "mean_woman":  float(w.mean()),
            "delta":       delta,
            "p_value":     float(p),
            "significant": p < 0.05,
            "cohens_d":    delta / pooled if pooled > 0 else 0.0,
        }
    return out


def save_stats_csv(stat: dict, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "mean_man", "mean_woman", "delta", "cohens_d",
                    "p_value", "significant"])
        for key, s in stat.items():
            w.writerow([key, f"{s['mean_man']:.4f}", f"{s['mean_woman']:.4f}",
                        f"{s['delta']:+.4f}", f"{s['cohens_d']:+.4f}",
                        f"{s['p_value']:.4f}", s["significant"]])
    print(f"  [Saved] {out_path}")


def plot_delta_bar(stat: dict, title: str, out_path: str) -> None:
    keys   = sorted(stat.keys(), key=lambda k: abs(stat[k]["delta"]), reverse=True)
    deltas = [stat[k]["delta"] for k in keys]
    colors = ["lightgray" if not stat[k]["significant"]
              else (C_MAN_BAR if stat[k]["delta"] > 0 else C_WOMAN_BAR)
              for k in keys]
    fig, ax = plt.subplots(figsize=(9, max(4, len(keys) * 0.55)))
    y_pos = np.arange(len(keys))
    ax.barh(y_pos, deltas, color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, k in enumerate(keys):
        m = _sig_marker(stat[k]["p_value"])
        if m:
            ax.text(deltas[i] + (0.02 if deltas[i] >= 0 else -0.02), i, m, va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_xlabel("Delta (Man − Woman)")
    ax.set_title(title, fontsize=10)
    handles = [mpatches.Patch(color=C_MAN_BAR,   alpha=0.9, label="Man > Woman (sig.)"),
               mpatches.Patch(color=C_WOMAN_BAR, alpha=0.9, label="Woman > Man (sig.)"),
               mpatches.Patch(color="lightgray",  alpha=0.9, label="Not significant")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def plot_mean_comparison(stat: dict, title: str, out_path: str) -> None:
    keys        = list(stat.keys())
    man_means   = [stat[k]["mean_man"]   for k in keys]
    woman_means = [stat[k]["mean_woman"] for k in keys]
    x = np.arange(len(keys))
    h = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    ax.bar(x - h/2, man_means,   h, label="Man (A)",   color=C_MAN_BAR,   alpha=0.85)
    ax.bar(x + h/2, woman_means, h, label="Woman (B)", color=C_WOMAN_BAR, alpha=0.85)
    y_max = max(max(man_means), max(woman_means))
    for i, k in enumerate(keys):
        m = _sig_marker(stat[k]["p_value"])
        if m:
            ax.text(x[i], y_max * 1.03, m, ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(_label(k), 14) for k in keys], fontsize=8)
    ax.set_ylabel("Mean Score")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, y_max * 1.15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Univariate AUC ─────────────────────────────────────────────────────────────

def compute_univariate_auc(score_keys, man_scores, woman_scores) -> dict[str, float]:
    y    = np.array([1] * len(man_scores) + [0] * len(woman_scores))
    aucs = {}
    for i, key in enumerate(score_keys):
        x = np.concatenate([man_scores[:, i], woman_scores[:, i]])
        try:
            auc = roc_auc_score(y, x)
        except Exception:
            auc = float("nan")
        aucs[key] = max(auc, 1 - auc)
    return aucs


def save_univariate_auc_csv(aucs: dict, stat: dict, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "univariate_auc", "cohens_d", "delta", "p_value", "significant"])
        for k in sorted(aucs, key=lambda k: aucs[k], reverse=True):
            s = stat[k]
            w.writerow([k, f"{aucs[k]:.4f}", f"{s['cohens_d']:+.4f}",
                        f"{s['delta']:+.4f}", f"{s['p_value']:.4f}", s["significant"]])
    print(f"  [Saved] {out_path}")


def plot_univariate_auc(aucs: dict, stat: dict, title: str, out_path: str) -> None:
    keys   = sorted(aucs, key=lambda k: aucs[k], reverse=True)
    vals   = [aucs[k] for k in keys]
    colors = ["lightgray" if not stat[k]["significant"]
              else (C_MAN_BAR if stat[k]["cohens_d"] > 0 else C_WOMAN_BAR)
              for k in keys]
    fig, ax = plt.subplots(figsize=(9, max(4, len(keys) * 0.55)))
    y_pos = np.arange(len(keys))
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.axvline(0.5, color="black", linewidth=1.0, linestyle="--")
    for i, (k, v) in enumerate(zip(keys, vals)):
        ax.text(v + 0.002, i, f"d={stat[k]['cohens_d']:+.2f}", va="center", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_xlabel("Univariate AUC (man vs woman)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.45, max(vals) * 1.08 + 0.05)
    handles = [mpatches.Patch(color=C_MAN_BAR,   alpha=0.85, label="man > woman"),
               mpatches.Patch(color=C_WOMAN_BAR, alpha=0.85, label="woman > man"),
               mpatches.Patch(color="lightgray",  alpha=0.85, label="not significant")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Logistic probing ───────────────────────────────────────────────────────────

def compute_logistic(score_keys, man_scores, woman_scores):
    """
    Logistic regression (female=1, male=0), 80/20 pair-level split.
    Returns: weights {dim: coef}, metrics dict, probs_all array
    """
    n = len(man_scores)
    tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    Xtr = np.vstack([man_scores[tr], woman_scores[tr]])
    ytr = np.array([0]*len(tr) + [1]*len(tr))
    Xte = np.vstack([man_scores[te], woman_scores[te]])
    yte = np.array([0]*len(te) + [1]*len(te))

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s  = sc.transform(Xte)

    clf = LogisticRegression(max_iter=10000, C=1.0, solver="lbfgs")
    clf.fit(Xtr_s, ytr)

    p_tr = clf.predict_proba(Xtr_s)[:, 1]
    p_te = clf.predict_proba(Xte_s)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p_te)

    X_all_s  = sc.transform(np.vstack([man_scores, woman_scores]))
    probs_all = clf.predict_proba(X_all_s)[:, 1]

    weights = dict(zip(score_keys, clf.coef_[0]))
    metrics = {
        "auc_train": roc_auc_score(ytr, p_tr),
        "acc_train": accuracy_score(ytr, clf.predict(Xtr_s)),
        "auc_test":  roc_auc_score(yte, p_te),
        "acc_test":  accuracy_score(yte, clf.predict(Xte_s)),
        "fpr": fpr, "tpr": tpr,
    }
    return weights, metrics, probs_all


def save_logistic_csv(weights: dict, metrics: dict, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "logistic_coef_std", "direction",
                    "auc_train", "acc_train", "auc_test", "acc_test"])
        for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            w.writerow([k, f"{v:.6f}", "woman" if v >= 0 else "man",
                        f"{metrics['auc_train']:.4f}", f"{metrics['acc_train']:.4f}",
                        f"{metrics['auc_test']:.4f}",  f"{metrics['acc_test']:.4f}"])
    print(f"  [Saved] {out_path}")


def _compute_pair_losses(probs: np.ndarray, n_pairs: int):
    eps = 1e-9
    pm  = probs[:n_pairs]
    pf  = probs[n_pairs:]
    la  = -np.log(1 - pm + eps)
    lb  = -np.log(pf + eps)
    return (la + lb) / 2


def _load_coef_csv(path: str) -> dict[str, float]:
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["dimension"]] = float(row["logistic_coef_std"])
    return result


# ── Per-model analysis ─────────────────────────────────────────────────────────

def run_single(label: str, path: str, out_dir: str, remove_outliers: int = 200) -> dict:
    """Run per-model analysis. Always fits logistic on COMMON_DIMS (cherry10)."""
    os.makedirs(out_dir, exist_ok=True)
    records = load_records(path)
    print(f"\n[{label}] {len(records)} records loaded")

    score_keys, man_scores, woman_scores = extract_scores(records)
    stat = compute_stats(score_keys, man_scores, woman_scores)

    save_stats_csv(stat, os.path.join(out_dir, "stats.csv"))
    plot_delta_bar(stat, f"Score Delta (Man − Woman) [{label}]",
                   os.path.join(out_dir, "delta_bar.png"))
    plot_mean_comparison(stat, f"Man vs Woman — Mean Scores [{label}]",
                         os.path.join(out_dir, "mean_comparison.png"))

    uni = compute_univariate_auc(score_keys, man_scores, woman_scores)
    save_univariate_auc_csv(uni, stat, os.path.join(out_dir, "univariate_auc.csv"))
    plot_univariate_auc(uni, stat, f"Univariate AUC [{label}]",
                        os.path.join(out_dir, "univariate_auc.png"))

    # Always fit on COMMON_DIMS (10 dims)
    idx     = [score_keys.index(k) for k in COMMON_DIMS if k in score_keys]
    c_keys  = [score_keys[i] for i in idx]
    c_man   = man_scores[:, idx]
    c_woman = woman_scores[:, idx]
    n_pairs = len(records)

    w1, m1, probs1 = compute_logistic(c_keys, c_man, c_woman)
    save_logistic_csv(w1, m1, os.path.join(out_dir, "logistic_weights_cherry10.csv"))
    print(f"  [Logistic] AUC_train={m1['auc_train']:.4f}  "
          f"AUC_test={m1['auc_test']:.4f}  Acc_test={m1['acc_test']:.4f}")

    # Console stats summary
    print(f"\n  {'Dimension':<38} {'Man':>6} {'Woman':>6} {'Delta':>8} {'d':>8} {'p':>8}")
    print(f"  {'-'*78}")
    for k, s in stat.items():
        sig = "*" if s["significant"] else ""
        print(f"  {k:<38} {s['mean_man']:>6.3f} {s['mean_woman']:>6.3f}"
              f" {s['delta']:>+8.3f} {s['cohens_d']:>+8.3f} {s['p_value']:>8.4f}{sig}")

    roc_entry = {
        "label":         label,
        "n_cherry":      len(c_keys),
        "fpr":           m1["fpr"],
        "tpr":           m1["tpr"],
        "auc_train":     m1["auc_train"],
        "auc_test":      m1["auc_test"],
        "acc_train":     m1["acc_train"],
        "acc_test":      m1["acc_test"],
        "fpr_clean": None, "tpr_clean": None,
        "auc_test_clean": None, "acc_test_clean": None,
    }

    # Outlier removal + re-fit
    actual_rm = min(remove_outliers, max(0, n_pairs - 10))
    if actual_rm > 0:
        pair_losses = _compute_pair_losses(probs1, n_pairs)
        keep = [i for i in range(n_pairs)
                if i not in set(np.argsort(pair_losses)[-actual_rm:])]
        print(f"\n  [Outlier removal] top-{actual_rm} pairs removed → {len(keep)} remain")

        w2, m2, _ = compute_logistic(c_keys, c_man[keep], c_woman[keep])
        save_logistic_csv(w2, m2,
                          os.path.join(out_dir, f"logistic_weights_cherry10_rm_outlier{actual_rm}.csv"))
        print(f"  [Re-fit]  AUC_train={m2['auc_train']:.4f}  "
              f"AUC_test={m2['auc_test']:.4f}  Acc_test={m2['acc_test']:.4f}")

        roc_entry["fpr_clean"]       = m2["fpr"]
        roc_entry["tpr_clean"]       = m2["tpr"]
        roc_entry["auc_test_clean"]  = m2["auc_test"]
        roc_entry["acc_test_clean"]  = m2["acc_test"]
        roc_entry["acc_train_clean"] = m2["acc_train"]

    return roc_entry


# ── Response-only ROC combined figures ─────────────────────────────────────────

def _make_roc_2x2(roc_entries: list[dict], out_path: str) -> None:
    """2×2 subplot, one panel per model. Shows original + cleaned ROC."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for i, entry in enumerate(roc_entries):
        ax    = axes[i]
        color = PALETTE[i]
        ax.plot(entry["fpr"], entry["tpr"], color=color, lw=2,
                label=f"Test ROC  (AUC={entry['auc_test']:.3f}   Acc={entry['acc_test']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        if entry.get("fpr_clean") is not None:
            ax.plot(entry["fpr_clean"], entry["tpr_clean"], color=color, lw=2,
                    linestyle="--",
                    label=f"Cleaned ROC (AUC={entry['auc_test_clean']:.3f}   "
                          f"Acc={entry['acc_test_clean']:.3f})")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.set_title(f"{entry['label']}  [cherry{entry['n_cherry']}]",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        info = [
            f"AUC  train={entry['auc_train']:.3f}  test={entry['auc_test']:.3f}",
            f"Acc  train={entry['acc_train']:.3f}  test={entry['acc_test']:.3f}",
        ]
        if entry.get("auc_test_clean") is not None:
            info += ["─── after outlier removal ───",
                     f"AUC_test={entry['auc_test_clean']:.3f}  "
                     f"Acc_test={entry['acc_test_clean']:.3f}"]
        ax.text(0.98, 0.38, "\n".join(info), transform=ax.transAxes,
                fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    for j in range(len(roc_entries), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("ROC Curves — Response-Only Eval (test set, 80/20 split)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def _make_roc_overlay(roc_entries: list[dict], out_path: str) -> None:
    """All 4 models overlaid, solid = original, dashed = cleaned."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random")
    for i, entry in enumerate(roc_entries):
        color = PALETTE[i]
        ax.plot(entry["fpr"], entry["tpr"], color=color, lw=2,
                label=f"{entry['label']} [ch{entry['n_cherry']}]  AUC={entry['auc_test']:.3f}")
        if entry.get("fpr_clean") is not None:
            ax.plot(entry["fpr_clean"], entry["tpr_clean"], color=color, lw=1.5,
                    linestyle="--",
                    label=f"  └ cleaned  AUC={entry['auc_test_clean']:.3f}  "
                          f"Acc={entry['acc_test_clean']:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def _make_roc_cleaned_only(roc_entries: list[dict], out_path: str) -> None:
    """All 4 models overlaid, cleaned ROC only."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random")
    for i, entry in enumerate(roc_entries):
        if entry.get("fpr_clean") is None:
            continue
        color = PALETTE[i]
        ax.plot(entry["fpr_clean"], entry["tpr_clean"], color=color, lw=2,
                label=(f"{entry['label']} [ch{entry['n_cherry']}]\n"
                       f"  AUC={entry['auc_test_clean']:.3f}  "
                       f"Acc={entry['acc_test_clean']:.3f}"))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves (Response-Only, Cleaned) — LLM Gender Bias\n"
                 "(top-200 high-loss pairs removed, test set)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def make_response_only_roc_figures(roc_entries: list[dict], out_dir: str) -> None:
    print("\n[ROC] Generating response-only combined ROC figures ...")
    _make_roc_2x2(roc_entries,
                  os.path.join(out_dir, "roc_response_only_all.png"))
    _make_roc_overlay(roc_entries,
                      os.path.join(out_dir, "roc_response_only_all_combined.png"))
    _make_roc_cleaned_only(roc_entries,
                           os.path.join(out_dir, "roc_response_only_rm_outliers.png"))


# ── Paper figures ──────────────────────────────────────────────────────────────

def _load_jsonl_paper(model_id: str, score_dir: str):
    """Load JSONL → (score_keys, man_scores, woman_scores, records)."""
    path = os.path.join(score_dir, f"score_response_{model_id}.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get("scores")
            if not isinstance(s, dict):
                continue
            if all(isinstance(v, dict) and "score_a" in v and "score_b" in v
                   and not v.get("error") for v in s.values()):
                records.append(rec)
    sk = list(records[0]["scores"].keys())
    mr, wr = [], []
    for rec in records:
        mr.append([float(rec["scores"][k]["score_a"]) for k in sk])
        wr.append([float(rec["scores"][k]["score_b"]) for k in sk])
    return sk, np.array(mr), np.array(wr), records


def _run_all_dims(score_keys, man_scores, woman_scores):
    """Logistic on ALL dims — used for metrics table, KDE, ROC paper figures."""
    w, m, probs = compute_logistic(score_keys, man_scores, woman_scores)
    n = len(man_scores)
    return {
        "probs_male":   probs[:n],
        "probs_female": probs[n:],
        "auc_test":     m["auc_test"],
        "acc_test":     m["acc_test"],
        "fpr":          m["fpr"],
        "tpr":          m["tpr"],
    }


# Model legend labels with line breaks for readability at large font sizes
_LEGEND_LABEL = {
    "GPT-5.4-mini":      "GPT-5.4-mini",
    "Claude-Sonnet-4.5": "Claude-\nSonnet-4.5",
    "Gemini-2.5-Flash":  "Gemini-\n2.5-Flash",
    "LLaMA-3.3-70B":     "LLaMA-\n3.3-70B",
}


# ── Paper Figure 1: delta_bar_std ─────────────────────────────────────────────

def _make_delta_bar_std(out_dir: str, score_dir_full: str) -> None:
    print("  [fig1] delta_bar_std.png ...")
    model_data = {}
    for mid, mname in BASE_MODELS:
        sk, ms, ws, _ = _load_jsonl_paper(mid, score_dir_full)
        model_data[mname] = {
            d: (float((ws[:, sk.index(d)] - ms[:, sk.index(d)]).mean()),
                float((ws[:, sk.index(d)] - ms[:, sk.index(d)]).std(ddof=1)))
            for d in COMMON_DIMS if d in sk
        }

    avg_d = {d: np.mean([model_data[mn].get(d, (0, 0))[0] for mn in model_data])
             for d in COMMON_DIMS}
    female_dims = sorted([d for d in COMMON_DIMS if avg_d[d] >= 0],
                         key=lambda d: avg_d[d], reverse=True)
    male_dims   = sorted([d for d in COMMON_DIMS if avg_d[d] < 0],
                         key=lambda d: avg_d[d])
    ordered = list(reversed(female_dims)) + list(reversed(male_dims))
    n_dims  = len(ordered)
    bar_h   = 0.18
    offsets = np.linspace(-3*bar_h/2, 3*bar_h/2, 4)

    fig, ax = plt.subplots(figsize=(36, max(30, n_dims * 5.0)))
    y_pos = np.arange(n_dims)
    for mi, ((mid, mname), color) in enumerate(zip(BASE_MODELS, PALETTE)):
        means = [model_data[mname].get(d, (0, 0))[0] for d in ordered]
        stds  = [model_data[mname].get(d, (0, 0))[1] for d in ordered]
        ax.barh(y_pos + offsets[mi], means, bar_h * 0.9,
                xerr=stds, color=color, alpha=0.82,
                error_kw=dict(elinewidth=2.0, capsize=6, ecolor="black", alpha=0.8),
                label=_LEGEND_LABEL.get(mname, mname))

    n_female = len(female_dims)
    if female_dims and male_dims:
        ax.axhline(n_female - 0.5, color="gray", lw=1.2, linestyle="--", alpha=0.6)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_slabel(d) for d in ordered], fontsize=84)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='x', labelsize=72)
    # Darken plot border
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")
    # Annotations inside the axes box
    ax.text(0.02, 0.99, "◀ Male > Female",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=72, color=C_MALE, style="italic", fontweight="bold")
    ax.text(0.98, 0.01, "Female > Male ▶",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=72, color=C_FEMALE, style="italic", fontweight="bold")
    ax.legend(fontsize=72, loc="upper right", framealpha=0.95,
              handlelength=1.2, labelspacing=1.0, borderpad=0.8,
              handletextpad=0.6)
    plt.tight_layout()
    out = os.path.join(out_dir, "delta_bar_std.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [Saved] {out}")


# ── Paper Figure 2: coef_all_models ───────────────────────────────────────────

def _make_coef(out_dir: str, score_dir_full: str) -> None:
    """
    Fits logistic regression on COEF_MALE_DIMS + COEF_FEMALE_DIMS (10 fixed dims),
    removes top-200 outlier pairs, and plots coefficients in the fixed paper order.
    """
    print("  [fig2] coef_all_models.png ...")
    all_coef_dims = COEF_MALE_DIMS + COEF_FEMALE_DIMS
    coef_data: dict[str, dict[str, float]] = {}

    for mid, mname in BASE_MODELS:
        sk, ms, ws, _ = _load_jsonl_paper(mid, score_dir_full)
        idx    = [sk.index(d) for d in all_coef_dims if d in sk]
        c_keys = [sk[i] for i in idx]
        c_man  = ms[:, idx]
        c_woman = ws[:, idx]

        w1, _, probs1 = compute_logistic(c_keys, c_man, c_woman)
        n_pairs = len(ms)
        pair_losses = _compute_pair_losses(probs1, n_pairs)
        rm   = min(200, max(0, n_pairs - 10))
        keep = [i for i in range(n_pairs)
                if i not in set(np.argsort(pair_losses)[-rm:])]
        w2, _, _ = compute_logistic(c_keys, c_man[keep], c_woman[keep])
        coef_data[mname] = w2

    if not coef_data:
        print("    [SKIP] No data loaded.")
        return

    model_names = [mn for _, mn in BASE_MODELS if mn in coef_data]

    # Fixed ordering: female at bottom (y=0…n_female-1), male at top (y=n_female…n_dims-1)
    # reversed() so #1 appears at the top of each section in barh
    ordered  = list(reversed(COEF_FEMALE_DIMS)) + list(reversed(COEF_MALE_DIMS))
    n_female = len(COEF_FEMALE_DIMS)
    n_dims   = len(ordered)

    male_rank   = {d: i+1 for i, d in enumerate(COEF_MALE_DIMS)}
    female_rank = {d: i+1 for i, d in enumerate(COEF_FEMALE_DIMS)}

    bar_h   = 0.18
    offsets = np.linspace(-3*bar_h/2, 3*bar_h/2, len(model_names))

    fig, ax = plt.subplots(figsize=(36, max(30, n_dims * 5.0)))
    y_pos = np.arange(n_dims)

    for mi, (name, color) in enumerate(zip(model_names, PALETTE)):
        for i, d in enumerate(ordered):
            v = coef_data[name].get(d, np.nan)
            if np.isnan(v):
                continue
            ax.barh(y_pos[i] + offsets[mi], v, bar_h * 0.92, color=color, alpha=0.85)
            ax.text(v + (0.15 if v >= 0 else -0.15), y_pos[i] + offsets[mi],
                    f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=30, color=color)

    ax.axhline(n_female - 0.5, color="gray", lw=1.2, linestyle="--", alpha=0.7)
    ax.axvline(0, color="black", lw=1.2)

    # Ensure x-axis range is wide enough to show all ticks (e.g. -4)
    xl = ax.get_xlim()
    ax.set_xlim(left=min(xl[0], -4.2), right=max(xl[1], 4.2))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"#{male_rank.get(d) or female_rank.get(d)}  {_slabel(d)}" for d in ordered],
        fontsize=78)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='x', labelsize=72)

    # Darken plot border
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")

    # Annotations inside the axes box
    ax.text(0.02, 0.99, "◀ Male > Female",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=72, color=C_MALE, style="italic", fontweight="bold")
    ax.text(0.98, 0.01, "Female > Male ▶",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=72, color=C_FEMALE, style="italic", fontweight="bold")

    handles = [mpatches.Patch(color=PALETTE[i], alpha=0.85,
                              label=_LEGEND_LABEL.get(n, n))
               for i, n in enumerate(model_names)]
    ax.legend(handles=handles, fontsize=72, loc="upper right",
              framealpha=0.95, handlelength=1.2, labelspacing=1.0,
              borderpad=0.8, handletextpad=0.6)
    plt.tight_layout()
    out = os.path.join(out_dir, "coef_all_models.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [Saved] {out}")


# ── Paper Figure 3: prob_kde_response_only ────────────────────────────────────

def _make_prob_kde(out_dir: str, score_dir_ro: str) -> None:
    print("  [fig3] prob_kde_response_only.png ...")
    fig, ax = plt.subplots(figsize=(13, 10))
    x_grid  = np.linspace(-0.02, 1.02, 500)
    handles = []
    for (mid, mname), color, ls in zip(RO_MODELS, PALETTE, LINESTYLES):
        sk, ms, ws, _ = _load_jsonl_paper(mid, score_dir_ro)
        res = _run_all_dims(sk, ms, ws)          # ALL dims
        n   = len(res["probs_male"])
        y_m = gaussian_kde(res["probs_male"],   bw_method=0.12)(x_grid) * n
        y_f = gaussian_kde(res["probs_female"], bw_method=0.12)(x_grid) * n
        ax.plot(x_grid, y_m, color=C_MALE,   lw=3, linestyle=ls)
        ax.plot(x_grid, y_f, color=C_FEMALE, lw=3, linestyle=ls)
        ax.fill_between(x_grid, y_m, alpha=0.08, color=C_MALE)
        ax.fill_between(x_grid, y_f, alpha=0.08, color=C_FEMALE)
        handles.append(plt.Line2D([0], [0], color="gray", lw=3, linestyle=ls, label=mname))

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(bottom=0)
    ax.set_xlabel("Classifier Score", fontsize=36)
    ax.set_ylabel("Count",           fontsize=36)
    ax.tick_params(axis='both', labelsize=30)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.text(0,   -0.10, "Male",   transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=30, fontweight="bold", color=C_MALE)
    ax.text(1.0, -0.10, "Female", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=30, fontweight="bold", color=C_FEMALE)
    ax.axvline(0, color=C_MALE,   lw=1.2, linestyle=":", alpha=0.5)
    ax.axvline(1, color=C_FEMALE, lw=1.2, linestyle=":", alpha=0.5)
    ax.legend(handles=handles, fontsize=30, title="Model", title_fontsize=28,
              framealpha=0.93, loc="upper center", bbox_to_anchor=(0.50, 0.99))
    ax.set_title("Score Distribution by Prompt Condition", fontsize=34, pad=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "prob_kde_response_only.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [Saved] {out}")


# ── Paper Figure 4: metrics_table ─────────────────────────────────────────────

def _make_metrics_table(out_dir: str, score_dir_full: str, score_dir_ro: str) -> None:
    print("  [fig4] metrics_table.png ...")
    rows = []
    for (bid, mname), (rid, _) in zip(BASE_MODELS, RO_MODELS):
        sk_f, ms_f, ws_f, _ = _load_jsonl_paper(bid, score_dir_full)
        rf = _run_all_dims(sk_f, ms_f, ws_f)    # ALL 20 dims

        sk_r, ms_r, ws_r, _ = _load_jsonl_paper(rid, score_dir_ro)
        rr = _run_all_dims(sk_r, ms_r, ws_r)    # ALL 20 dims

        rows.append({"model":    mname,
                     "auc_full": rf["auc_test"], "auc_ro": rr["auc_test"],
                     "acc_full": rf["acc_test"], "acc_ro": rr["acc_test"]})

    col_labels = ["Model", "AUC\n(full)", "AUC\n(response-only)",
                  "Acc\n(full)", "Acc\n(response-only)"]
    col_keys   = ["auc_full", "auc_ro", "acc_full", "acc_ro"]
    col_rank   = {}
    for key in col_keys:
        vals = sorted(set(r[key] for r in rows), reverse=True)
        col_rank[key] = {vals[0]: 1}
        if len(vals) > 1:
            col_rank[key][vals[1]] = 2

    import matplotlib as mpl
    prev = mpl.rcParams['font.family']
    mpl.rcParams['font.family'] = 'serif'
    n_r, n_c = len(rows), len(col_labels)
    fig, ax  = plt.subplots(figsize=(38, 1.7*(n_r+1)+0.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cw = [0.26, 0.185, 0.185, 0.185, 0.185]
    cx = [sum(cw[:i]) for i in range(n_c)]
    tr = [1.0 - i/(n_r+1) for i in range(n_r+2)]

    def cy(ri): return (tr[ri] + tr[ri+1]) / 2
    def cc(ci): return cx[ci] + cw[ci] / 2
    def hl(y, lw, a=1.0):
        ax.plot([0, 1], [y, y], color="black", lw=lw, alpha=a,
                transform=ax.transAxes, clip_on=False)

    hl(tr[0], 3.5); hl(tr[1], 2.0); hl(tr[-1], 3.5)
    for ri in range(2, n_r+1):
        hl(tr[ri], 0.7, 0.35)
    for ci, lbl in enumerate(col_labels):
        ax.text(cc(ci), cy(0), lbl, ha="center", va="center",
                fontsize=48, fontweight="bold", transform=ax.transAxes)
    for ri, r in enumerate(rows):
        cells = [r["model"]] + [f"{r[k]:.4f}" for k in col_keys]
        for ci, (key, txt) in enumerate(zip([None]+col_keys, cells)):
            fw, fi = "normal", "normal"
            if key:
                rk = col_rank[key].get(r[key], 99)
                if rk == 1: fw = "bold"
                elif rk == 2: fi = "italic"
            ax.text(cc(ci), cy(ri+1), txt, ha="center", va="center",
                    fontsize=50, fontweight=fw, fontstyle=fi,
                    transform=ax.transAxes)
    plt.tight_layout(pad=0.3)
    mpl.rcParams['font.family'] = prev

    out = os.path.join(out_dir, "metrics_table.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"    [Saved] {out}")

    print(f"\n  Metrics (all dims):")
    for r in rows:
        print(f"    {r['model']:<26} AUC(full)={r['auc_full']:.4f}  "
              f"AUC(ro)={r['auc_ro']:.4f}  "
              f"Acc(full)={r['acc_full']:.4f}  Acc(ro)={r['acc_ro']:.4f}")


# ── Paper Figure 5: roc_response_only ────────────────────────────────────────

def _run_coef_dims(score_keys, man_scores, woman_scores):
    """Logistic on COEF_MALE_DIMS + COEF_FEMALE_DIMS (10 fixed dims)."""
    all_coef_dims = COEF_MALE_DIMS + COEF_FEMALE_DIMS
    idx    = [score_keys.index(d) for d in all_coef_dims if d in score_keys]
    c_keys = [score_keys[i] for i in idx]
    c_man  = man_scores[:, idx]
    c_woman = woman_scores[:, idx]
    w, m, probs = compute_logistic(c_keys, c_man, c_woman)
    n = len(c_man)
    return {
        "probs_male":   probs[:n],
        "probs_female": probs[n:],
        "auc_test":     m["auc_test"],
        "acc_test":     m["acc_test"],
        "fpr":          m["fpr"],
        "tpr":          m["tpr"],
    }


def _make_roc_paper(out_dir: str, score_dir_ro: str) -> None:
    print("  [fig5] roc_response_only.png ...")
    fig, ax = plt.subplots(figsize=(22, 20))
    ax.plot([0, 1], [0, 1], "k--", lw=2.0, alpha=0.4, label="Random (AUC=0.5)")
    for (mid, mname), color in zip(RO_MODELS, PALETTE):
        sk, ms, ws, _ = _load_jsonl_paper(mid, score_dir_ro)
        res = _run_all_dims(sk, ms, ws)            # ALL 20 dims
        ax.plot(res["fpr"], res["tpr"], color=color, lw=7.0,
                label=f"{mname}\nAUC={res['auc_test']:.3f}   Acc={res['acc_test']:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=72)
    ax.set_ylabel("True Positive Rate",  fontsize=72)
    ax.tick_params(axis='both', labelsize=60)
    ax.legend(fontsize=44, loc="lower right", framealpha=0.95,
              handlelength=2.0, labelspacing=1.0, borderpad=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")
    plt.tight_layout()
    out = os.path.join(out_dir, "roc_response_only.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [Saved] {out}")


def make_paper_figures(out_dir: str) -> None:
    score_full = os.path.join(REPO_ROOT, "results", "scores", "full")
    score_ro   = os.path.join(REPO_ROOT, "results", "scores", "response_only")
    os.makedirs(out_dir, exist_ok=True)
    print("\n[ANALYSIS] Generating paper figures ...")
    _make_delta_bar_std(out_dir, score_full)
    _make_coef(out_dir, score_full)
    _make_prob_kde(out_dir, score_ro)
    _make_metrics_table(out_dir, score_full, score_ro)
    _make_roc_paper(out_dir, score_ro)
    print(f"  [Done] All 5 paper figures saved to {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRAP gender-bias statistical analysis")
    parser.add_argument("--score_dir",
                        default=os.path.join(REPO_ROOT, "results", "scores", "full"),
                        help="Directory with score_*.jsonl files (default: results/scores/full)")
    parser.add_argument("--out_dir",
                        default=os.path.join(REPO_ROOT, "results", "analysis"),
                        help="Output directory (default: results/analysis)")
    parser.add_argument("--remove_outliers", type=int, default=200,
                        help="Remove top-N high-loss pairs before re-fitting (default: 200)")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip paper figure generation")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.score_dir, "score_*.jsonl")))
    if not files:
        print(f"[ERROR] No score_*.jsonl files found in {args.score_dir}")
        return

    is_ro = "response_only" in args.score_dir

    print(f"[ANALYSIS] Found {len(files)} file(s) in {args.score_dir}:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    def _label_from(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return name.removeprefix("score_response_")

    roc_list = []
    for path in files:
        label   = _label_from(path)
        out_dir = os.path.join(args.out_dir, label)
        entry   = run_single(label, path, out_dir,
                             remove_outliers=args.remove_outliers)
        roc_list.append(entry)

    if not args.no_figures:
        if is_ro:
            # Response-only: generate combined ROC figures
            make_response_only_roc_figures(roc_list, args.out_dir)
        else:
            # Full condition: generate paper figures
            make_paper_figures(args.out_dir)

    print(f"\n[ANALYSIS] Done → {args.out_dir}")


if __name__ == "__main__":
    main()
