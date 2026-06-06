"""Download dataset files from the Hugging Face Hub, with SCRATCH_DIR-based caching."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def hf_download(repo_id: str, filename: str, revision: Optional[str] = None) -> str:
    """Download *filename* from *repo_id* and return its local path.

    Files are cached under ``$SCRATCH_DIR/``.  Repeated calls with the same arguments skip the download.
    """
    from huggingface_hub import hf_hub_download

    local_dir: Optional[str] = None
    scratch = os.getenv("SCRATCH_DIR")
    if scratch:
        repo_name = repo_id.split("/")[-1]
        local_dir = os.path.join(scratch, repo_name)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type="dataset",
        local_dir=local_dir,
    )
    logger.info("HF dataset %s:%s -> %s", repo_id, filename, local_path)
    return local_path
