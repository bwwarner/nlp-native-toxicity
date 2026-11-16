# Loads all three models behind the one shared scoring interface
# Owner: Garrett (the baseline loader follows Brant's baseline.joblib; adjust if his signature differs)
# Run: imported
# Reads: A/models/baseline.joblib, A/models/distilbert/, A/models/distilbert_aug/
#        or the HF Hub repos in env vars HF_DISTILBERT_REPO / HF_DISTILBERT_AUG_REPO
# Writes: nothing
#
#     score_fn(texts: list[str]) -> list[float]   # P(toxic) in [0, 1], toxic if >= 0.5
import os

from src.common import MODELS_DIR

MODEL_NAMES = ["baseline", "distilbert", "distilbert_aug"]
HF_REPOS = {
    "distilbert": os.environ.get("HF_DISTILBERT_REPO"),
    "distilbert_aug": os.environ.get("HF_DISTILBERT_AUG_REPO"),
}

def load_baseline(path=MODELS_DIR / "baseline.joblib"):
    import joblib

    obj = joblib.load(path)
    if hasattr(obj, "predict_proba"):  # sklearn Pipeline(TfidfVectorizer, LogisticRegression)
        return lambda texts: obj.predict_proba(list(texts))[:, 1].tolist()
    if isinstance(obj, dict) and {"vectorizer", "model"} <= obj.keys():
        return lambda texts: obj["model"].predict_proba(obj["vectorizer"].transform(list(texts)))[:, 1].tolist()
    raise TypeError(f"Don't know how to score with {type(obj)} from {path}")


def load(name):
    if name == "baseline":
        return load_baseline()
    from src.distilbert_scorer import load_scorer

    local = MODELS_DIR / name
    src = str(local) if (local / "config.json").exists() else HF_REPOS.get(name)
    if not src:
        raise FileNotFoundError(f"No local {local} and no HF repo env var for {name}")
    return load_scorer(src)


def load_available(names=MODEL_NAMES):
    """Load every model that exists; skip (with a note) the ones that don't yet."""
    out = {}
    for n in names:
        try:
            out[n] = load(n)
        except Exception as e:  # missing model shouldn't block auditing the others
            print(f"[skip] {n}: {e}")
    return out
