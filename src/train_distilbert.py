# Fine-tunes the DistilBERT on the training split and evaluates it
# Owner: Garrett
# Run: python -m src.train_distilbert [--aug] [--sample 50000] [--epochs 1]
# Read: A/splits/train, val, test .parquet  (columns: text, label)
#       --aug: A/splits/train_aug.parquet   (from python -m src.augment)
# Writes: A/models/distilbert[_aug]/            (model + tokenizer, loadable with from_pretrained)
#         A/preds/distilbert[_aug]_test.csv      (columns: text, label, score)
#         results/distilbert[_aug]_metrics.csv   (precision, recall, f1, roc_auc on val and test)
#
# Settings: distilbert-base-uncased, max_len 128, lr 2e-5, batch 32, 1 epoch,
# weight decay 0.01, 6% warmup, seed 42. Best checkpoint chosen by val F1, so
# test is only used once at the end.
#
# --sample N trains on a stratified sample of N original training comments
# (the proposal's 50k fallback if the full 200k is too slow on Colab).
# With --aug the SAME N originals are drawn (same seed), plus their counterfactual
# copies, so the plain and augmented models differ only by augmentation.
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.common import MODELS_DIR, PREDS_DIR, RESULTS_DIR, SEED, SPLITS_DIR, THRESHOLD
from src.distilbert_scorer import MAX_LEN

BASE_MODEL = "distilbert-base-uncased"
ID2LABEL = {0: "non_toxic", 1: "toxic"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def metrics(labels, scores, threshold=THRESHOLD):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan"),
        "n": len(labels),
    }


def stratified_sample(df, n, seed=SEED):
    if n is None or n >= len(df):
        return df
    frac = n / len(df)
    return df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)


def load_train(aug, sample):
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    chosen = stratified_sample(train, sample)
    if not aug:
        return chosen[["text", "label"]]
    train_aug = pd.read_parquet(SPLITS_DIR / "train_aug.parquet")
    keep = train_aug[train_aug.source_idx.isin(set(chosen.index))]
    print(f"Augmented train: {(~keep.augmented).sum()} originals + {keep.augmented.sum()} counterfactuals")
    return keep[["text", "label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true", help="train on train_aug.parquet")
    ap.add_argument("--sample", type=int, default=None, help="use N original train comments (e.g. 50000)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--base", default=BASE_MODEL, help="base checkpoint (tiny model for smoke tests)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (e.g. a Mac where MPS crashes)")
    ap.add_argument("--max_eval", type=int, default=None, help="cap val/test rows (smoke tests only)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    name = "distilbert_aug" if args.aug else "distilbert"
    out_dir = MODELS_DIR / name
    for d in (out_dir, PREDS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df = load_train(args.aug, args.sample)
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")[["text", "label"]]
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")[["text", "label"]]
    if args.max_eval:
        val_df, test_df = val_df.head(args.max_eval), test_df.head(args.max_eval)
    print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)} | toxic rate {train_df.label.mean():.3f}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    def to_ds(df):
        ds = Dataset.from_pandas(df.reset_index(drop=True).rename(columns={"label": "labels"}))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                      remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    def compute_metrics(p):
        logits, labels = p
        scores = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return {k: v for k, v in metrics(labels, scores).items() if k != "n"}

    targs = TrainingArguments(
        output_dir=str(MODELS_DIR.parent / "checkpoints" / name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.06 * math.ceil(len(train_ds) / args.batch) * args.epochs),  # 6% warmup
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",
        use_cpu=args.cpu,
        seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    rows = []
    for split, ds, df in (("val", val_ds, val_df), ("test", test_ds, test_df)):
        logits = trainer.predict(ds).predictions
        scores = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        rows.append({"model": name, "split": split, **metrics(df.label.values, scores)})
        if split == "test":
            pd.DataFrame({"text": df.text.values, "label": df.label.values, "score": scores}).to_csv(
                PREDS_DIR / f"{name}_test.csv", index=False)
    m = pd.DataFrame(rows)
    m.to_csv(RESULTS_DIR / f"{name}_metrics.csv", index=False)
    print(m.to_string(index=False))
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
