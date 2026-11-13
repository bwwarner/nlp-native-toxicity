# Garrett → Brant handoff

Status key: ✅ done in repo · ⏳ waiting on a training run · 🤝 needs both of us

## 1. Identity terms ✅ — `data/identity_terms.json`

```json
{"native": ["Native American", "Indigenous", "American Indian", "First Nations",
            "Chickasaw", "Navajo", "Cherokee", "Choctaw", "Lakota", "Ojibwe"],
 "other":  ["Black", "white", "Asian", "Latino", "Muslim",
            "Christian", "Jewish", "gay", "Irish", "Mexican"]}
```

- Exactly two keys, `native` and `other`. Lists of display-case strings.
- Terms are used as **adjectives** ("a Navajo woman"), so any one can be swapped for any other.
- Match **whole words, case-insensitively, longest first** ("Native American" before anything shorter). `src/common.py` does this already, so the app can reuse it:

```python
from src.common import load_terms, find_terms, swap_term, swap_all
swap_all("I am a proud Chickasaw woman.", score_fn)  # -> list of {term, group, text, score, delta}
```

`swap_term` also fixes a/an ("a Navajo" → "an Irish") and capitalization at the start of a sentence.

## 2. Scoring interface 🤝 (please confirm or send me yours)

Every model, baseline included:

```python
score_fn(texts: list[str]) -> list[float]   # P(toxic) in [0, 1]
label = "toxic" if score >= 0.5 else "non-toxic"
```

- DistilBERT: `src/distilbert_scorer.load_scorer(path_or_repo)` returns this function.
- Baseline: `src/scorers.load_baseline()` assumes `baseline.joblib` is a sklearn `Pipeline` (TF-IDF → LogisticRegression) with `predict_proba`, **or** a dict `{"vectorizer", "model"}`. If you save it differently, tell me and I'll adjust the loader, or just export a `score(texts)` from `src/baseline.py` with the signature above.
- `src/scorers.load_available()` returns `{"baseline": fn, "distilbert": fn, "distilbert_aug": fn}` and skips any model that doesn't exist yet. The app can use it directly.

## 3. Model repos ⏳

| Model | Hub repo (after `python -m src.push_to_hub --user <me>`) | Local path |
|---|---|---|
| DistilBERT | `<garrett-hf-user>/native-toxicity-distilbert` | `A/models/distilbert/` |
| DistilBERT + aug | `<garrett-hf-user>/native-toxicity-distilbert-aug` | `A/models/distilbert_aug/` |

I'll fill in the username and say whether the repos are public. If they're private, I'll add an `HF_TOKEN` secret to the Space instead of pasting it anywhere.

How to load them in the Space (must match training exactly):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok   = AutoTokenizer.from_pretrained(REPO)            # distilbert-base-uncased tokenizer, saved with the model
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
enc   = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
score = model(**enc).logits.softmax(-1)[:, 1]          # label order: 0 = non_toxic, 1 = toxic
```

Or: set `HF_DISTILBERT_REPO` / `HF_DISTILBERT_AUG_REPO` env vars and call `src.scorers.load_available()`.

## 4. Demo examples ⏳

Candidates are in `data/demo_cases.json` (5 representative, 4 failure, 4 ambiguous). After training I run `python -m src.demo_cases`, which writes `results/demo_cases.csv` (score + label for each model, plus native-minus-other gap) and `results/demo_cases_swaps.csv`. Then I send you the final 5–6 for `gr.Examples`:

- 2–3 representative: a harmless Native sentence with the biggest native-minus-other gap
- 1 failure: a toxic anti-Native sentence that a model scores < 0.5
- 1 ambiguous: "tribe"/"reservation" in a non-identity sense

**Final list:** _TBD after `demo_cases` runs._

## 5. How swap-and-rescore works (we both need to explain this)

1. Find the first identity term in the input (whole-word, case-insensitive).
2. Rewrite the sentence once for every other term in the list, changing nothing else (a/an fixed).
3. Score every version with the same model.
4. Show each version's score and its **delta** from the original. If the sentence is harmless and only the identity word changed, a fair model gives about the same score for every version. A consistent positive gap for Native terms is the bias we're measuring.
5. This is the per-sentence version of the counterfactual token fairness gap in Garg et al. (2019). `src/audit.py` computes it over the 60 neutral and 60 toxic templates.

Limitation: it's keyword-based, so "Jeep Cherokee" or "white wine" get swapped too, and terms outside the list ("Indians", "Natives", "rez") aren't detected.

## 6. The models, in one paragraph each

- **Baseline (Brant):** TF-IDF turns each comment into a sparse vector of word weights (frequent in this comment × rare in the corpus). Logistic regression learns one weight per word and outputs a sigmoid probability. No word order or context, so "not stupid" looks like "stupid". Its upside is that the weights can be inspected: if "gay" or "Navajo" has a positive weight, that's direct evidence the term itself pushes toward toxic.
- **DistilBERT (Garrett):** a 6-layer transformer distilled from BERT-base (Sanh et al., 2019), about 40% smaller and 60% faster with most of BERT's accuracy. Pretrained `distilbert-base-uncased` plus a 2-class head, fine-tuned for 1 epoch on our Civil Comments train split (lr 2e-5, batch 32, max 128 tokens, seed 42). The best checkpoint is picked by val F1, and test is only scored once. Because it reads words in context, it should handle negation and sarcasm better, but it can also absorb associations between identity words and toxicity from the pretraining and fine-tuning data.
- **DistilBERT + augmentation (Garrett):** identical setup, except the training set adds k = 2 counterfactual copies of every comment that mentions an identity term, with that term swapped (at least one swap crosses groups, so Native terms appear in many non-toxic contexts). The label stays the same, which tells the model the identity word shouldn't change the decision. Only train is augmented, never val or test. On the 50k fallback, both models use the same 50k original comments.

## 7. Still to do together 🤝

- [ ] Brant confirms the baseline score signature (handoff #2)
- [ ] Decide who drives the live demo and rehearse the slide → demo transitions
- [ ] Slide 5 points to the demo, and slide 2 mentions augmentation (or that we dropped it)
- [ ] Space link + video link in the README and the Canvas upload, **by Sunday 1pm**
- [ ] Brant records the backup video **after** the final demo cases are fixed
- [ ] Each person logs their own AI use in `docs/genai_log.md`
