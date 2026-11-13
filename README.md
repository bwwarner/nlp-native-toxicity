# nlp-native-toxicity
Repo for Brant and Garrett's CS72 Final Project on NLP native toxicity

## Project Structure
```
GitHub repo (code + small files)        Shared Drive folder (big artifacts)
nlp-native-toxicity/                    toxicity-artifacts/
├── README.md                           ├── splits/
├── requirements.txt                    │   ├── train.parquet
├── .gitignore                          │   ├── val.parquet
├── src/                                │   └── test.parquet
│   ├── config.py   (paths, seed)       ├── models/
│   ├── data.py                         │   ├── baseline.joblib
│   ├── evaluate.py                     │   ├── distilbert/
│   ├── baseline.py                     │   └── distilbert_aug/
│   ├── train_distilbert.py             └── preds/
│   ├── augment.py                          ├── baseline_test.csv
│   ├── templates.py                        ├── distilbert_test.csv
│   └── audit.py                            └── ...
├── app.py                              
├── colab/run_pipeline.ipynb  (shell cells only)
├── data/   templates.csv, identity_terms.json
├── results/  metrics CSVs, figures
└── docs/   genai_log.md
```

#### Environment Variable:
- ARTIFACT_DIR is where splits, models, and preds (predictions) will be stored. Locally default to ./artifacts

#### Shared Conventions:
- random seed will always be 42
- label threshold will always be 0.5
- the prediction file will have columns text, label, and score