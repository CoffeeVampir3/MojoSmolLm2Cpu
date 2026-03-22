# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub"]
# ///
"""Download SmolLM2-135M weights and tokenizer to checkpoints/SmolLM2/."""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DIR = PROJECT_ROOT / "checkpoints" / "SmolLM2"

snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(LOCAL_DIR),
    allow_patterns=["model.safetensors", "tokenizer.json", "config.json"],
)

print(f"Downloaded to {LOCAL_DIR}")
