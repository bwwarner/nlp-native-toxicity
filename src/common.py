# Shared helpers for Garrett's files: paths, identity terms, and term matching/swapping
# Owner: Garrett
# Run: imported
# Reads: src/config.py (if filled in), env var ARTIFACT_DIR, data/identity_terms.json
# Writes: nothing
#
# The values below fall back to the README's shared conventions until Brant
# fills in src/config.py. Once he does, they come from there.
import json
import os
import re
from pathlib import Path

from src import config

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(getattr(config, "ARTIFACT_DIR", os.environ.get("ARTIFACT_DIR", REPO_ROOT / "artifacts")))
SEED = getattr(config, "SEED", 42)
THRESHOLD = getattr(config, "THRESHOLD", 0.5)

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
SPLITS_DIR = ARTIFACT_DIR / "splits"
MODELS_DIR = ARTIFACT_DIR / "models"
PREDS_DIR = ARTIFACT_DIR / "preds"
TERMS_PATH = DATA_DIR / "identity_terms.json"


def load_terms(path=TERMS_PATH):
    """Return {"native": [...], "other": [...]}."""
    with open(path) as f:
        return json.load(f)


def all_terms(terms=None):
    terms = terms or load_terms()
    return terms["native"] + terms["other"]


def group_of(term, terms=None):
    terms = terms or load_terms()
    return "native" if term in terms["native"] else "other"


def term_regex(terms):
    """Whole-word, case-insensitive matcher. Longest terms first so
    "Native American" wins over a shorter overlapping term."""
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in ordered) + r")\b", re.IGNORECASE)


def find_terms(text, terms=None, regex=None):
    """Return the canonical identity terms that appear in text (in order, no dupes)."""
    terms = terms or all_terms()
    regex = regex or term_regex(terms)
    lookup = {t.lower(): t for t in terms}
    found = []
    for m in regex.finditer(text):
        t = lookup[m.group(0).lower()]
        if t not in found:
            found.append(t)
    return found


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def swap_term(text, old, new):
    """Replace every whole-word occurrence of `old` with `new` (case-insensitive),
    fixing a/an right in front of it. This is the core of swap-and-rescore and of
    counterfactual augmentation."""
    pattern = re.compile(r"\b(?:(a|an)(\s+))?(" + re.escape(old) + r")\b", re.IGNORECASE)

    def repl(m):
        det, space, word = m.group(1), m.group(2), m.group(3)
        # Keep sentence-initial capitalization for lowercase terms like "gay"/"white".
        rep = new[0].upper() + new[1:] if word[0].isupper() and m.start(3) == 0 else new
        if det is None:
            return rep
        a = article(rep)
        return (a.capitalize() if det[0].isupper() else a) + space + rep

    return pattern.sub(repl, text)


def swap_all(text, score_fn, terms=None):
    """Swap-and-rescore. Finds the first identity term in `text`, rewrites the
    sentence with every other term, and scores each version.

    Returns a list of dicts: term, group, text, score, delta (score minus the
    original sentence's score). The original row has delta 0.
    """
    terms = terms or load_terms()
    found = find_terms(text, all_terms(terms))
    if not found:
        return []
    original = found[0]
    variants = [(original, text)] + [
        (t, swap_term(text, original, t)) for t in all_terms(terms) if t.lower() != original.lower()
    ]
    scores = score_fn([v for _, v in variants])
    base = scores[0]
    return [
        {"term": t, "group": group_of(t, terms), "text": v, "score": float(s), "delta": float(s - base)}
        for (t, v), s in zip(variants, scores)
    ]
