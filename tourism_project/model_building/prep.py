"""
Stage 2 of the MLOps pipeline: clean the raw data and build the train and test splits.

Reads the registered raw dataset from the Hugging Face Hub, applies the cleaning
rules identified during exploratory analysis, produces a stratified 80/20 split,
and uploads Xtrain, Xtest, ytrain and ytest back to the dataset repository.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "AhmedBenHamouda"
# ------------------------------------------------------------------------

DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "tourism_project/data"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN is not set.")

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------- load
# Pull the registered raw file rather than a local copy, so this stage is
# reproducible on any machine including a fresh CI runner.
local_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(local_path)
print(f"Loaded raw dataset from the Hub: {df.shape[0]} rows, {df.shape[1]} columns")

# --------------------------------------------------------------- clean
# 1. Drop the export index column and the unique identifier. Neither carries signal.
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# 2. Merge the inconsistent category labels found during exploratory analysis.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 3. Impute any missing values defensively. The supplied file is complete, but a
#    refreshed extract may not be, and the pipeline re-runs on every push.
for col in df.columns:
    if df[col].isna().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  imputed {col} with the mode")
        else:
            df[col] = df[col].fillna(df[col].median())
            print(f"  imputed {col} with the median")

# 4. Cap the extreme tails instead of deleting rows. The dataset is small and
#    imbalanced, so losing positive cases would cost more than the outliers do.
for col in ["DurationOfPitch", "NumberOfTrips", "MonthlyIncome"]:
    low, high = df[col].quantile([0.01, 0.99])
    n_capped = int(((df[col] < low) | (df[col] > high)).sum())
    df[col] = df[col].clip(low, high)
    print(f"  capped {col} to [{low:.0f}, {high:.0f}], {n_capped} values adjusted")

# 5. Drop exact duplicate records. Once CustomerID is removed, some rows become
#    identical across all 19 columns. Leaving them in is not neutral: an identical
#    row landing in both the train and the test split leaks the answer and inflates
#    the reported test score.
before = len(df)
df = df.drop_duplicates()
print(f"  removed {before - len(df)} duplicate rows (identical once CustomerID is dropped)")

print(f"Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Positive class rate: {df[TARGET].mean() * 100:.2f}%")

# --------------------------------------------------------------- split
X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,          # preserves the 19.3% positive rate in both splits
)
print(f"Train: {Xtrain.shape}, positive rate {ytrain.mean() * 100:.2f}%")
print(f"Test:  {Xtest.shape}, positive rate {ytest.mean() * 100:.2f}%")

# --------------------------------------------------------------- save
os.makedirs(OUTPUT_DIR, exist_ok=True)
splits = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}

for filename, frame in splits.items():
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=filename,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add prepared split {filename}",
    )
    print(f"Uploaded {filename} to {DATASET_REPO}")

print("Data preparation complete.")
