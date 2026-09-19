"""
Stage 1 of the MLOps pipeline: register the raw dataset on the Hugging Face Hub.

Uploads tourism_project/data/tourism.csv to a versioned dataset repository so that
every downstream stage reads from one remote source of truth.
"""

import os
import sys
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "AhmedBenHamouda"
# ------------------------------------------------------------------------

DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"   # created automatically
LOCAL_FILE = "tourism_project/data/tourism.csv"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN is not set. Export it locally or add it to your GitHub secrets.")

api = HfApi(token=HF_TOKEN)

# Create the dataset repository if it does not exist. This keeps the script
# idempotent, which matters because CI re-runs it on every push.
try:
    api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Dataset repository already exists: {DATASET_REPO}")
except RepositoryNotFoundError:
    create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, token=HF_TOKEN)
    print(f"Created dataset repository: {DATASET_REPO}")

if not os.path.exists(LOCAL_FILE):
    sys.exit(f"Raw data file not found at {LOCAL_FILE}")

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Register raw tourism dataset",
)

print(f"Raw dataset registered at https://huggingface.co/datasets/{DATASET_REPO}")
