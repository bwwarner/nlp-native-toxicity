# Downloads Civil Comments, dedupes, labels at 0.5, samples 200k, and splits
# Owner: Brant
# Run: python -m src.data
# Reads: google/civil-comments
# Writes: A/splits/{train,val,test}.parquet

import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src import config


# load the dataset
# it gives us train val, and test splits
# convert each set to pandas and only keep text and toxicity
def load_split(name, limit=None):
    ds = load_dataset("google/civil_comments", split=name)
    if limit:
        ds = ds.select(range(limit))

    df = ds.to_pandas()[["text", "toxicity"]]

    return df

# binary label
# is it toxic?
def add_label(df):
    df = df.copy()
    df["label"] = (df["toxicity"] >= config.THRESHOLD).astype(int)

    return df

# dedupe and prevent leakage
# drop dupe text values in training split
# prevent leakage by removing from val and test any text that appears in train
def dedupe_and_leakage(train, val, test):

    # dedupe
    train = train.drop_duplicates(subset="text")
    val = val.drop_duplicates(subset="text")
    test = test.drop_duplicates(subset="text")

    # remove val rows that appear in training
    train_texts = set(train["text"])
    val = val[~val["text"].isin(train_texts)].copy()

    # remove test rows that appear in training and val
    val_texts = set(val["text"].dropna())
    test_texts = train_texts | val_texts
    test = test[~test["text"].isin(test_texts)].copy()

    return train, val, test

# sample the training set
# take about 200k rows with fixed seed stratified on label
# this preserves toxic portion
def sample_train(train):
    if len(train) <= config.TRAIN_SAMPLE_SIZE:
        return train # prevents errors from being raised when we do __limit runs with far fewer rows

    sampled, _ = train_test_split(train,
                                  train_size=config.TRAIN_SAMPLE_SIZE,
                                  stratify=train["label"], # keeps toxic proportion the same
                                  random_state=config.SEED)

    return sampled.reset_index(drop=True)

# save as parquet
def save_splits(train, val, test):
    config.ensure_dirs()

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df[["text", "label", "toxicity"]].reset_index(drop=True).to_parquet(
            config.SPLITS_DIR / f"{name}.parquet", index=False
        )

# summary printout
def summarize(train, val, test):
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s} rows={len(df):>9,} toxic={df['label'].mean():.2%}")

# code driver
# also contains limit argument
# limit takes only the first N rows of each split for small splice test mode
# allows us to test without downloading and processing all millions of rows
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Using only the first N rows of each split for testing")

    args = parser.parse_args()

    train = load_split("train", limit=args.limit)
    val = load_split("validation", limit=args.limit)
    test = load_split("test", limit=args.limit)

    train, val, test = (add_label(d) for d in (train, val, test))
    train, val, test = dedupe_and_leakage(train, val, test)
    train = sample_train(train)
    save_splits(train, val, test)
    summarize(train, val, test)

# run
if __name__ == "__main__":
    main()