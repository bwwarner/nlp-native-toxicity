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
- Training sample size will be 200,000

## Demo & Video
- Live demo (Hugging Face Space): https://huggingface.co/spaces/bwwarner/nlp-native-toxicity
- Backup video: https://drive.google.com/file/d/1830dVpMoQAcfeESwVSJ2ixWH3m9l_Z7m/view?usp=sharing

## Reproducing the DistilBERT + audit pipeline (Garrett)
Run on a GPU (Colab: `colab/run_distilbert.ipynb` does all of this). Needs Brant's splits in `$ARTIFACT_DIR/splits/` first.

```bash
pip install -r requirements.txt
python -m src.templates                                   # data/templates.csv (2400 sentences)
python -m src.augment                                     # A/splits/train_aug.parquet
python -m src.train_distilbert --sample 50000             # A/models/distilbert/
python -m src.train_distilbert --aug --sample 50000       # A/models/distilbert_aug/
python -m src.audit                                       # results/ metrics, errors, figures
python -m src.demo_cases                                  # results/demo_cases.csv
python -m src.push_to_hub --user <hf-username>           # weights -> HF Hub for the Space
```

Drop `--sample` to train on the full ~200k split. Use the same value for both runs.
Scoring interface shared by all models: `score_fn(texts: list[str]) -> list[float]` (P(toxic), toxic if >= 0.5).
See `docs/handoff_garrett.md` for model loading details and the swap-and-rescore explanation.
