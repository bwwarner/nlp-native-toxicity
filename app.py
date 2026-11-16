# Gradio Demo that scores a comments w/ a chosen model and shows the score shift under identity-term swaps
# Owner: Brant
# Run: python app.py
# Read: models from HF Hub (or A/models/
# Writes: nothing (this serves as the UI for the demo)

import gradio as gr
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import re
import pandas as pd

THRESHOLD = 0.5 # borrowed from config.py
MAX_LEN = 128

# model names and repos
BASELINE = "Baseline (TF-IDF + LR)"
DISTIL_REPOS = {
    "DistilBERT": "bwwarner/native-toxicity-distilbert",
    "DistilBERT + Augmentation": "bwwarner/native-toxicity-distilbert-aug"
}
MODEL_CHOICES = [BASELINE, *DISTIL_REPOS]

# load model once
def load_model():
    path = hf_hub_download(
        repo_id="bwwarner/nlp-native-toxicity-models",
        filename = "baseline.joblib",
    )
    return joblib.load(path)

# time for latency testing
import time
t0 = time.time()
MODEL = load_model()
print(f"Model loaded in {time.time()-t0:.3f} seconds")

# load the DistilBERT models once at startup
t0 = time.time()
MODELS = {BASELINE: MODEL}
for name, repo in DISTIL_REPOS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()
    MODELS[name] = (tok, mdl)

print(f"DistilBERT models loaded in {time.time()-t0:.3f} seconds")

# shared model interface with toxic proba for a list of texts for any model
def score_texts(model_name, texts):
    m = MODELS[model_name]
    if model_name == BASELINE:
        return m.predict_proba(texts)[:, 1].tolist() # baseline case

    tok, mdl = m
    out = []

    for i in range (0, len(texts), 32):
        enc = tok(texts[i:i+32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        with torch.no_grad():
            out += torch.softmax(mdl(**enc).logits, dim=-1)[:, 1].tolist()

    return out

# identity terms (same 20 as the audit)
TERMS = {
    "native": ["Native American", "Indignous", "American Indian", "First Nations",
               "Chicksaw", "Navajo", "Cherokee", "Choctaw", "Lakota", "Ojibwe"],
    "other": ["Black", "white", "Asian", "Latino", "Muslim", "Christian", "Jewish",
              "gay", "Irish", "Mexican"]
}
ALL_TERMS = sorted([(t, g) for g, ts in TERMS.items() for t in ts],
                   key = lambda x: -len(x[0])) # longest one first
GROUP = {t.lower(): g for t, g in ALL_TERMS}
PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t, _ in ALL_TERMS) + r")\b", re.I)
EMPTY = pd.DataFrame(columns=["Term", "Group", "Score", "Shift vs. Original", "Label"])

# swap detected term for every other term and rescore in one batch
def swap_table(text, model_name, base_score):
    m = PATTERN.search(text)
    if not m:
        return EMPTY, "No identity term from our list was found, so there is no swap table"
    found = m.group(0)
    swaps = [t for t, _ in ALL_TERMS if t.lower() != found.lower()]
    variants = [text[:m.start()] + t + text[m.end():] for t in swaps]
    scores = score_texts(model_name, variants)
    rows = [{"Term": found + " (original)", "Group": GROUP[found.lower()],
             "Score": round(base_score, 3), "Shift vs. Original": 0.0,
             "Label": "Toxic" if base_score >= THRESHOLD else "Not toxic"}]

    for t, s in zip(swaps, scores):
        rows.append({"Term": t, "Group": GROUP[t.lower()], "Score": round(s, 3),
                     "Shift vs. Original": round(s - base_score, 3),
                     "Label": "Toxic" if s >= THRESHOLD else "Not toxic"})

    return pd.DataFrame(rows), f"Detected **{found}**. Each row swaps it for another term and rescores."

# examples for testing, along with their corresponding model
EXAMPLES = [
    ["White people drink too much.", BASELINE],
    ["I am a proud Chickasaw woman.", BASELINE],
    ["The Cherokee council voted on a new school.", BASELINE],
    ["Go back to the reservation where you belong.", BASELINE],
    ["Those Indians should have been wiped out.", BASELINE],
    ["Those Indians should have been wiped out.", "DistilBERT"],
    ["Natives are all drunks living off government handouts.", "DistilBERT"],
    ["Natives are all drunks living off government handouts.", "DistilBERT + Augmentation"],
    ["Our marketing tribe is meeting at noon to plan the launch.", BASELINE],
    ["He drove his Jeep Cherokee to the lake.", BASELINE],
    ["You are not stupid", BASELINE]
]

# gradio handler
def handler(text, model_name):
    t = time.time()

    if not text or not text.strip():
        return None, "Please enter some text.", "", EMPTY

    text = text.strip()[:2000]
    score = score_texts(model_name, [text])[0]

    label = "Toxic" if score >= THRESHOLD else "Not toxic"

    table, note = swap_table(text, model_name, score)

    print(f"Request: {time.time()-t:.3f} seconds")

    return round(score, 3), label, note, table

# gradio interface
demo = gr.Interface(
    fn = handler,
    inputs=[
        gr.Textbox(label="Enter a comment", placeholder="Type something here..."),
        gr.Dropdown(MODEL_CHOICES, value=BASELINE, label="Model")
    ],
    outputs=[
        gr.Number(label="Toxicity Score"),
        gr.Textbox(label="Prediction"),
        gr.Markdown(),
        gr.Dataframe(label="Identity-term swap")
    ],
    title="Does Toxicity Detection Penalize Native Identity?",
    description=("TRIGGER WARNING: Example phrases contain offensive/toxic language, which is unfortunately a requirement "
                 "for the tests we are conducting.\n This webpage scores user-entered comments using one of three models, then swaps the identity term "
    "and rescores. Limitations: Civil Comments has no Native identity labels, "
    "and swaps only work for the 20 listed terms, not words like \"Indians\" or \"Natives\"."),
    examples = EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()