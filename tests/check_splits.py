import pandas as pd
from src import config

train = pd.read_parquet(config.SPLITS_DIR / "train.parquet")
val = pd.read_parquet(config.SPLITS_DIR / "val.parquet")
test = pd.read_parquet(config.SPLITS_DIR / "test.parquet")

for name, df in [("train", train), ("val", val), ("test", test)]:
    print(name, len(df), f"{df['label'].mean():.2%} toxic")
    assert list(df.columns) == ["text", "label", "toxicity"], df.columns
    assert df["text"].notna().all(), "null text"
    assert df["text"].is_unique, f"duplicate text in {name}"
    assert set(df["label"].unique()) <= {0, 1}
    assert ((df["toxicity"] >= config.THRESHOLD).astype(int) == df["label"]).all(), "label/threshold mismatch"

tr, va, te = set(train["text"]), set(val["text"]), set(test["text"])
assert not (tr & va), "train/val leakage"
assert not (tr & te), "train/test leakage"
assert not (va & te), "val/test overlap"
print("All checks passed")