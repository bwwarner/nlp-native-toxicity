# Gradio Demo that scores a comments w/ a chosen model and shows the score shift under identity-term swaps
# Owner: Brant
# Run: python app.py
# Read: models from HF Hub (or A/models/
#       data.identity_terms.json
# Writes: nothing (this serves as the UI for the demo)

import gradio as gr
import joblib

from src import config

# load model once
def load_model():
    return joblib.load(config.MODELS_DIR / "baseline.joblib")

MODEL = load_model()

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
        "Toxic" if score >= config.THRESHOLD
        else "Not toxic"
    )
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
    description="TF-IDF + Logistic regresion toxicity classifier"
)

if __name__ == "__main__":
    demo.launch()