# Runs every candidate demo input through all three models, plus swap-and-rescore
# Owner: Garrett
# Run: python -m src.demo_cases
# Read: data/demo_cases.json, data/identity_terms.json, models (see src/scorers.py)
# Writes: results/demo_cases.csv        one row per case: score + label per model
#         results/demo_cases_swaps.csv  per case x model x swapped term: score, delta
#
# Use the output to pick the final gr.Examples list for Brant:
#   - representative: harmless Native sentence whose score is clearly higher than
#     the same sentence with "other" terms (look at native_minus_other)
#   - failure: toxic anti-Native sentence that a model scores < 0.5
#   - ambiguous: "tribe"/"reservation" in a non-identity sense
import json

import pandas as pd

from src import scorers
from src.common import DATA_DIR, RESULTS_DIR, THRESHOLD, load_terms, swap_all


def main():
    cases = json.load(open(DATA_DIR / "demo_cases.json"))["cases"]
    terms = load_terms()
    models = scorers.load_available()
    rows, swaps = [], []
    for c in cases:
        row = {"kind": c["kind"], "text": c["text"]}
        for m, fn in models.items():
            s = fn([c["text"]])[0]
            row[f"{m}_score"] = round(s, 3)
            row[f"{m}_label"] = "toxic" if s >= THRESHOLD else "non-toxic"
            res = swap_all(c["text"], fn, terms)
            if res:
                df = pd.DataFrame(res)
                gap = df[df.group == "native"].score.mean() - df[df.group == "other"].score.mean()
                row[f"{m}_native_minus_other"] = round(gap, 3)
                swaps += [{"case": c["text"], "model": m, **r} for r in res]
        rows.append(row)
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "demo_cases.csv", index=False)
    pd.DataFrame(swaps).to_csv(RESULTS_DIR / "demo_cases_swaps.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
