"""
Model pre-download script for Docker build stage.
Downloads the model weights into /models/age-gender so the resulting image is self-contained.
"""
import os
import sys
from huggingface_hub import snapshot_download

MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
TARGET_DIR = os.environ.get("MODEL_DIR", "/models/age-gender")

print(f"Downloading model {MODEL_ID} to {TARGET_DIR}...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
)
print(f"Model successfully saved to {TARGET_DIR}!")
