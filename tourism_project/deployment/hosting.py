"""
Stage 4 of the MLOps pipeline: publish the application to a Hugging Face Space.

Uploads the Space folder (app.py, requirements.txt) to the Gradio Space, which then
rebuilds and restarts the application serving the registered model.
"""

import os
import sys
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "AhmedBenHamouda"
# ------------------------------------------------------------------------

SPACE_REPO = f"{HF_USERNAME}/tourism-package-prediction"
DEPLOY_DIR = "tourism_project/deployment/space"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN is not set.")

api = HfApi(token=HF_TOKEN)

# The Space is normally created from the web interface so the hardware can be set
# to ZeroGPU. This fallback keeps CI from failing if it does not exist yet.
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space already exists: {SPACE_REPO}")
except RepositoryNotFoundError:
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="gradio",
                private=False, token=HF_TOKEN)
    print(f"Created Space: {SPACE_REPO}")

# Upload the folder in a single commit so the Space rebuilds once.
# README.md is deliberately not uploaded, so the sdk_version set when the Space
# was created is preserved.
api.upload_folder(
    folder_path=DEPLOY_DIR,
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
    commit_message="Deploy tourism package prediction app",
)

print(f"Application deployed to https://huggingface.co/spaces/{SPACE_REPO}")
print("The Space rebuilds on every upload, which takes a couple of minutes.")
