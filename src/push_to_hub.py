# Uploads the fine-tuned DistilBERT folders to the Hugging Face Hub so the Gradio Space can load them
# Owner: Garrett
# Run: huggingface-cli login   (or set HF_TOKEN), then
#      python -m src.push_to_hub --user <your-hf-username> [--private]
# Read: A/models/distilbert/, A/models/distilbert_aug/
# Writes: <user>/native-toxicity-distilbert and <user>/native-toxicity-distilbert-aug on the Hub
import argparse

from huggingface_hub import HfApi

from src.common import MODELS_DIR

REPOS = {"distilbert": "native-toxicity-distilbert", "distilbert_aug": "native-toxicity-distilbert-aug"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    for local, repo in REPOS.items():
        folder = MODELS_DIR / local
        if not (folder / "config.json").exists():
            print(f"[skip] {folder} has no trained model")
            continue
        repo_id = f"{args.user}/{repo}"
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id,
                          commit_message="Upload fine-tuned DistilBERT (CS72 Native toxicity audit)")
        print(f"Uploaded {folder} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
