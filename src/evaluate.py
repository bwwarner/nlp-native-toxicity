# Computes F1, precision, recall, ROC-AUC, and subgroup metrics for the model's predictions
# Owner: Brant
# Run: imported
# Reads: text, label, score passed in by callers
# Writes: nothing

import pandas as pd
from src import config
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# returns a dict with F1, precision, recall, and ROC-AUC
# applies the threshold from config.py
# ROC-AUC is computed on raw scores
def compute_metrics(labels, scores):

    # apply threshold to get binary preds
    preds = (scores >= config.THRESHOLD).astype(int)

    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, preds)
    }

# groups by column then calls compute_metrics
def compute_group_metrics(df, group_col, labels_col, scores_col):
    results = {}

    for group, subset in df.groupby(group_col):
        results[group] = compute_metrics(
            subset[labels_col],
            subset[scores_col]
        )

    return results