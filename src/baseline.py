# Trains and evaluates the TF-IDF and logistic regression baseline
# Owner: Brant
# Run: python -m src.baseline
# Reads: A/splits/*
# Writes: A/models/baseline.joblib
#         A/preds/baseline_test.csv
#         results/baseline_metrics.csv

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.evaluate import compute_metrics

# load train and val splits from the artifacts folder
def load_train_val():
    train = pd.read_parquet(config.SPLITS_DIR / "train.parquet")
    val = pd.read_parquet(config.SPLITS_DIR / "val.parquet")
    return train, val

# load test split from the artifacts folder
# kept separate so test is not touched until very end
def load_test():
    return pd.read_parquet(config.SPLITS_DIR / "test.parquet")

# build the vectorizing and regression pipeline
def build_pipeline(C):

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range = (1, 2),
            min_df = 5,
            sublinear_tf = True
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced"
        )),
    ])

# tune model by fitting on training data, evaluating on al data nad returning best
def tune_model(train, val):
    best_model = None
    best_C = None
    best_f1 = -1

    for C in [0.1, 1.0, 10.0]:
        pipe = build_pipeline(C)

        # fit only using training data
        pipe.fit(train["text"], train["label"])

        # evaluate using val data
        val_scores = pipe.predict_proba(val["text"])[:, 1]
        metrics = compute_metrics(val["label"], val_scores)

        print(f"C={C}: {metrics}")

        # use better f1 to choose new model
        if metrics["f1"]  > best_f1:
            best_f1 = metrics["f1"]
            best_C = C
            best_model = pipe

    print(f"\nBest C: {best_C}")
    print(f"Best validation F1: {best_f1:.3f}")

    return best_model, best_C

# find the terms most associated with toxicity
# print those terms and the weights associated with that
def inspect_weights(pipe):
    features = pipe["tfidf"].get_feature_names_out()
    coeffs = pipe["clf"].coef_[0]

    weights = pd.DataFrame({
        "term": features,
        "weight": coeffs
    })

    # highest positive coeffs are most associated with toxic class
    top_toxic = weights.nlargest(20, "weight").copy()
    top_toxic["category"] = "top_toxic"

    # manually chosen identity-related terms for inspection
    identity_terms = ["indigenous", "gay", "muslim"]

    identity = weights[weights["term"].isin(identity_terms)].copy()
    identity["category"] = "identity_term"

    output = pd.concat([top_toxic, identity])
    output = output.drop_duplicates(subset="term")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # print the findings
    print("\nTop 20 toxic associated terms:")
    print(top_toxic[["term", "weight"]].to_string(index=False))

    print("\nIdentity term weights:")
    print(identity[["term", "weight"]].to_string(index=False))

    # save findings
    output.to_csv(
        config.RESULTS_DIR / "baseline_weights.csv",
        index = False
    )

    print("\nSaved baseline_weights.csv")

# code driver
def main():

    # output directories
    config.ensure_dirs()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # load data
    train, val = load_train_val()

    print("Loading dataset...")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")

    # tune only using only val to avoid leakage
    pipe, best_C = tune_model(train, val)

    # only load test after model is selected to avoid any model leakage
    test = load_test()
    print(f"Test: {len(test)}")

    # test set is evaluated only after model is selected to prevent leakage
    test_scores = pipe.predict_proba(test["text"])[:, 1]
    test_labels = (test_scores >= config.THRESHOLD).astype(int)

    metrics = compute_metrics(test["label"], test_scores)

    # print metrics
    print("\nFinal Test Metrics:")
    print(metrics)

    # save test metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        config.RESULTS_DIR / "baseline_metrics.csv",
        index = False
    )

    # save test preds
    preds = pd.DataFrame({
        "text": test["text"],
        "label": test_labels,
        "socre": test_scores
    })

    preds.to_csv(
        config.PREDS_DIR / "baseline_test.csv",
        index = False
    )

    # save model pipeline
    joblib.dump(pipe, config.MODELS_DIR / "baseline.joblib")

    # inspect learned coeffs
    inspect_weights(pipe)

    print("\nBaseline Training Complete")

# run
if __name__ == "__main__":
    main()
