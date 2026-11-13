# Loads a fine-tuned DistilBERT (local folder or HF Hub repo) and returns a score function
# Owner: Garrett
# Run: imported  (quick check: python -m src.distilbert_scorer <path-or-repo> "some text")
# Reads: A/models/distilbert[_aug]/  or  <hf-user>/<repo> on the Hugging Face Hub
# Writes: nothing
#
# SHARED SCORING INTERFACE (all three models):
#     score_fn(texts: list[str]) -> list[float]
#     each float = P(toxic) in [0, 1]; label = toxic if score >= 0.5
#
# Model details Brant needs to match:
#     tokenizer   : the one saved with the model (distilbert-base-uncased)
#     max length  : 128 tokens, truncation on, padding to longest in batch
#     label order : index 0 = non_toxic, index 1 = toxic  -> score = softmax(logits)[:, 1]
import sys

MAX_LEN = 128


def load_scorer(name_or_path, device=None, batch_size=32, token=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        # MPS (Apple GPU) is skipped on purpose: it crashes with some macOS/torch combos.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name_or_path, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path, token=token).to(device).eval()

    @torch.no_grad()
    def score_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            out.extend(probs.float().cpu().tolist())
        return out

    return score_fn


if __name__ == "__main__":
    fn = load_scorer(sys.argv[1])
    for t, s in zip(sys.argv[2:], fn(sys.argv[2:])):
        print(f"{s:.3f}  {'TOXIC' if s >= 0.5 else 'ok   '}  {t}")
