# Dupes training comments containing identity terms, swaps in the other terms (including Native ones)
# Owner: Garrett
# Run: python -m src.augment [--k 2]
# Read: A/splits/train.parquet
#       data/identity_terms.json
# Writes: A/splits/train_aug.parquet  (original rows + counterfactual copies, shuffled;
#                                      extra columns: `augmented` marks the copies,
#                                      `source_idx` = row index in train.parquet it came from)
#
# Counterfactual data augmentation: for every training comment that mentions an
# identity term, we add k copies where that term is replaced by a different one.
# The copy keeps the original label, which teaches the model that the identity
# word alone should not change the toxicity decision. Each comment always gets at
# least one swap into the *opposite* group, so every comment mentioning e.g.
# "Muslim" also produces a copy mentioning a Native term. Native terms are rare in
# Civil Comments, so this is how the model sees them in many non-toxic contexts.
#
# Only train is augmented. Val and test are never touched (no leakage).
#
# Known limitation: matching is keyword-based, so non-identity uses ("white
# house", "black coffee", "the Navajo pickup truck") are also swapped.
import argparse
import random

import pandas as pd

from src.common import SEED, SPLITS_DIR, all_terms, find_terms, group_of, load_terms, swap_term, term_regex


def augment(df, terms, k=2, seed=SEED):
    rng = random.Random(seed)
    every = all_terms(terms)
    regex = term_regex(every)
    new_rows = []
    for idx, row in zip(df.index, df.itertuples(index=False)):
        found = find_terms(row.text, every, regex)
        if not found:
            continue
        original = found[0]
        opposite = terms["other"] if group_of(original, terms) == "native" else terms["native"]
        picks = [rng.choice(opposite)]
        pool = [t for t in every if t.lower() != original.lower() and t not in picks]
        picks += rng.sample(pool, k - 1) if k > 1 else []
        for new in picks:
            r = row._asdict()
            r["text"] = swap_term(row.text, original, new)
            r["augmented"] = True
            r["source_idx"] = idx
            new_rows.append(r)
    out = df.copy()
    out["augmented"] = False
    out["source_idx"] = df.index
    out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True), len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="counterfactual copies per matching comment")
    args = ap.parse_args()

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    terms = load_terms()
    aug, n_new = augment(train, terms, k=args.k)
    out = SPLITS_DIR / "train_aug.parquet"
    aug.to_parquet(out, index=False)
    n_match = n_new // args.k
    print(f"{n_match} of {len(train)} train comments mention an identity term ({n_match / len(train):.1%})")
    print(f"Added {n_new} counterfactual rows -> {len(aug)} total. Wrote {out}")
    print("Toxic rate before/after:", round(train.label.mean(), 4), round(aug.label.mean(), 4))


if __name__ == "__main__":
    main()
