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

THRESHOLD = 0.5 # borrowed from config.py
MAX_LEN = 512

DISTIL_REPOS = {
    ""
}

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

# examples for testing
EXAMPLES = [
    ["My grandmother taught me to speak Chicksaw."],
    ["My grandmother taught me to speak Irish."],
    ["The Navajo council voted on a new school."],
    ["The Muslim council voted on a new school."],
    ["Indigenous people deserve better than this."],
    ["The tribe held a meeting about the new reservation."],
    ["You are not stupid."],
    ["Wow, that sounds great."],
]

# shared model interface
def score_baseline(text) -> float:
    score = MODEL.predict_proba([text])[0, 1]
    return float(score)

# gradio handler
def handler(text):
    if not text or not text.strip():
        return None, "Please enter some text."

    score = score_baseline(text)

    label = (
        "Toxic" if score >= THRESHOLD
        else "Not toxic"
    )

    # track timing
    t = time.time()
    print(f"Request: {time.time()-t:.3f} seconds")

    return round(score, 3), label

# gradio interface
demo = gr.Interface(
    fn = handler,
    inputs=gr.Textbox(
        label="Enter a comment",
        placeholder="Type something here..."
    ),
    outputs=[
        gr.Number(label="Toxicity Score"),
        gr.Textbox(label="Prediction")
    ],
    title="Toxicity Detection Baseline",
    description="TF-IDF + Logistic regresion toxicity classifier",
    examples = EXAMPLES
)

if __name__ == "__main__":
    demo.launch()