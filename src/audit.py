# Scores templates w/ all three models and computes the score shift, false positive/negative rates, subgroup metrics, and top 50 errors
# Owner: Garrett
# Run: python -m src.audit
# Read: data/templates.csv
#       data/identity_terms.json
#       all models in A/models (or HF Hub, see src/scorers.py)
#       A/preds/{model}_test.csv  (text, label, score) for the real-data subgroup metrics
# Writes (results/):
#   template_scores.csv     every template row x model score
#   audit_by_term.csv       per model x term: mean neutral score, FPR, FNR, mean toxic score
#   audit_by_group.csv      per model x group (native / other): same, pooled
#   audit_shift.csv         counterfactual score shift (Garg et al. 2019) per model
#   subgroup_auc.csv        Borkan et al. 2019 subgroup / BPSN / BNSP AUC on the test split
#   errors_{model}_template_fp.csv / _fn.csv   50 highest-scored harmless Native templates
#                                              and 50 lowest-scored toxic Native templates
#   errors_{model}_test_fp.csv / _fn.csv       same on real test comments that mention Native terms
#                                              (blank `cause` column to hand-code by cause)
#   fig_neutral_by_term.png, fig_rates_by_group.png
#
# Definitions (threshold 0.5):
#   FPR  = share of neutral templates scored toxic   (harmless sentence flagged)
#   FNR  = share of toxic templates scored non-toxic (abuse missed)
#   shift (native - other): for each template, mean score over Native fills minus
#         mean over other fills, averaged across templates. > 0 means Native
#         mentions get higher toxicity for the same sentence.
#   CTF gap: for each template, mean |score(a) - score(b)| over all term pairs,
#         averaged across templates (0 = the term never matters).
#   subgroup AUC: AUC on test comments mentioning the group only.
#   BPSN AUC: non-toxic in-group vs toxic out-of-group (low = in-group false positives).
#   BNSP AUC: toxic in-group vs non-toxic out-of-group (low = in-group misses).
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import scorers
from src.common import DATA_DIR, PREDS_DIR, RESULTS_DIR, THRESHOLD, all_terms, find_terms, load_terms, term_regex

# Reference palette categorical slots 1-3 (validated all-pairs) + recessive ink
MODEL_COLORS = {"baseline": "#2a78d6", "distilbert": "#eb6834", "distilbert_aug": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def score_templates(models):
    df = pd.read_csv(DATA_DIR / "templates.csv")
    for name, fn in models.items():
        print(f"Scoring {len(df)} templates with {name}...")
        df[name] = fn(df.text.tolist())
    return df


def rates(sub, m):
    neu, tox = sub[sub.label == 0], sub[sub.label == 1]
    return {
        "mean_neutral_score": neu[m].mean(),
        "fpr": (neu[m] >= THRESHOLD).mean(),
        "mean_toxic_score": tox[m].mean(),
        "fnr": (tox[m] < THRESHOLD).mean(),
        "n_neutral": len(neu),
        "n_toxic": len(tox),
    }


def by_term_and_group(df, names):
    term_rows, group_rows = [], []
    for m in names:
        for (term, group), sub in df.groupby(["term", "group"], sort=False):
            term_rows.append({"model": m, "term": term, "group": group, **rates(sub, m)})
        for group, sub in df.groupby("group", sort=False):
            group_rows.append({"model": m, "group": group, **rates(sub, m)})
    return pd.DataFrame(term_rows), pd.DataFrame(group_rows)


def shift(df, names):
    rows = []
    for m in names:
        for label, sub in df.groupby("label"):
            diffs, gaps = [], []
            for _, t in sub.groupby("template_id"):
                g = t.groupby("group")[m].mean()
                diffs.append(g["native"] - g["other"])
                s = t[m].values
                gaps.append(np.mean([abs(a - b) for a, b in itertools.combinations(s, 2)]))
            rows.append({"model": m, "templates": "neutral" if label == 0 else "toxic",
                         "shift_native_minus_other": np.mean(diffs),
                         "shift_ci95": 1.96 * np.std(diffs, ddof=1) / np.sqrt(len(diffs)),
                         "ctf_gap": np.mean(gaps), "n_templates": len(diffs)})
    return pd.DataFrame(rows)


def _auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def subgroup_metrics(names, terms):
    """Borkan et al. (2019) on real test comments, using keyword membership."""
    rows, tagged = [], {}
    every = all_terms(terms)
    regex = term_regex(every)
    native = set(terms["native"])
    for m in names:
        path = PREDS_DIR / f"{m}_test.csv"
        if not path.exists():
            print(f"[skip] subgroup AUC for {m}: no {path}")
            continue
        p = pd.read_csv(path)
        found = p.text.astype(str).map(lambda t: find_terms(t, every, regex))
        p["is_native"] = found.map(lambda f: any(t in native for t in f))
        p["is_other"] = found.map(lambda f: any(t not in native for t in f))
        tagged[m] = p
        groups = {"native": p.is_native, "other": p.is_other}
        groups.update({t: found.map(lambda f, t=t: t in f) for t in every})
        for g, mask in groups.items():
            ins, out = p[mask], p[~mask]
            bpsn = pd.concat([ins[ins.label == 0], out[out.label == 1]])
            bnsp = pd.concat([ins[ins.label == 1], out[out.label == 0]])
            rows.append({
                "model": m, "subgroup": g, "n": len(ins), "n_toxic": int(ins.label.sum()),
                "subgroup_auc": _auc(ins.label, ins.score),
                "bpsn_auc": _auc(bpsn.label, bpsn.score),
                "bnsp_auc": _auc(bnsp.label, bnsp.score),
                "fpr": ((ins.score >= THRESHOLD) & (ins.label == 0)).sum() / max((ins.label == 0).sum(), 1),
                "fnr": ((ins.score < THRESHOLD) & (ins.label == 1)).sum() / max((ins.label == 1).sum(), 1),
            })
    return pd.DataFrame(rows), tagged


def write_errors(df, tagged, names, k=50):
    for m in names:
        nat = df[df.group == "native"]
        nat[nat.label == 0].nlargest(k, m)[["template_id", "text", "term", m]].assign(cause="").to_csv(
            RESULTS_DIR / f"errors_{m}_template_fp.csv", index=False)
        nat[nat.label == 1].nsmallest(k, m)[["template_id", "text", "term", m]].assign(cause="").to_csv(
            RESULTS_DIR / f"errors_{m}_template_fn.csv", index=False)
        if m in tagged:
            p = tagged[m][tagged[m].is_native]
            p[p.label == 0].nlargest(k, "score")[["text", "label", "score"]].assign(cause="").to_csv(
                RESULTS_DIR / f"errors_{m}_test_fp.csv", index=False)
            p[p.label == 1].nsmallest(k, "score")[["text", "label", "score"]].assign(cause="").to_csv(
                RESULTS_DIR / f"errors_{m}_test_fn.csv", index=False)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(by_term, by_group, names, terms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = terms["native"] + terms["other"]
    # 1) mean score on NEUTRAL templates, per term, per model
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    _style(ax)
    h = 0.8 / len(names)
    y = np.arange(len(order))
    for i, m in enumerate(names):
        vals = by_term[by_term.model == m].set_index("term").loc[order, "mean_neutral_score"]
        ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.9, color=MODEL_COLORS.get(m), label=m)
    ax.set_yticks(y, order, color=INK)
    ax.invert_yaxis()
    ax.axhline(len(terms["native"]) - 0.5, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("Mean toxicity score on harmless template sentences (lower is better)", color=INK2)
    fig.suptitle("Same harmless sentence, different identity term", color=INK, x=0.02, ha="left", fontsize=13)
    ax.set_title("Native terms above the dashed line, comparison terms below", color=INK2, loc="left", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=len(names),
               labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(RESULTS_DIR / "fig_neutral_by_term.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # 2) FPR / FNR by group, per model
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (col, title) in zip(axes, (("fpr", "Harmless flagged (FPR)"), ("fnr", "Abuse missed (FNR)"))):
        _style(ax)
        for i, m in enumerate(names):
            vals = by_group[by_group.model == m].set_index("group").loc[["native", "other"], col]
            ax.barh(np.arange(2) + i * h - 0.4 + h / 2, vals, height=h * 0.9, color=MODEL_COLORS.get(m), label=m)
        ax.set_yticks([0, 1], ["Native", "other"], color=INK)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlim(0, 1)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=len(names),
               labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS_DIR / "fig_rates_by_group.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    terms = load_terms()
    models = scorers.load_available()
    if not models:
        raise SystemExit("No models found. Train them first (see README).")
    names = list(models)

    df = score_templates(models)
    df.to_csv(RESULTS_DIR / "template_scores.csv", index=False)
    by_term, by_group = by_term_and_group(df, names)
    by_term.to_csv(RESULTS_DIR / "audit_by_term.csv", index=False)
    by_group.to_csv(RESULTS_DIR / "audit_by_group.csv", index=False)
    sh = shift(df, names)
    sh.to_csv(RESULTS_DIR / "audit_shift.csv", index=False)
    sub, tagged = subgroup_metrics(names, terms)
    if len(sub):
        sub.to_csv(RESULTS_DIR / "subgroup_auc.csv", index=False)
    write_errors(df, tagged, names)
    plot(by_term, by_group, names, terms)

    pd.set_option("display.width", 160)
    print("\n== By group (templates) ==\n", by_group.round(3).to_string(index=False))
    print("\n== Counterfactual shift ==\n", sh.round(4).to_string(index=False))
    if len(sub):
        print("\n== Subgroup AUC (test) ==\n",
              sub[sub.subgroup.isin(["native", "other"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
