# Fine-tunes the DistilBERT on the training split and evaluates it
# Owner: Garrett
# Run: python -m src.train_distilbert [--aug]
# Read: A/splits/train, val, test
#       (-- aug:
#       A/splits/train_aug.parquet
# Writes: A/models/distilbert[_aug]/
#         A/preds/distilbert[_aug]_test.csv
#         results/distilbert[_aug]_metrics.csv
