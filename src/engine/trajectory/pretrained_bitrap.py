"""Import audited official PIE NP weights without accepting arbitrary pickle."""

from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from .bitrap_adapter import BiTraPAdapter


OFFICIAL_PIE_WEIGHTS = {
    "912815ff732f20dc7ca190a49d2e82c11c1070016c799211e88517ba7192c5dc":
        (1, "1e6JmPbqK94E1Z--JExzklDvEjprFAyUs"),
    "858ce863fd5d34766323798db928e85297d0fb363e822f9be07cbb3de418decf":
        (20, "1jLkwi1YSwCRfixAxL6K5cNAvzs3NqtVJ"),
}


def import_pie_checkpoint(source, destination, config):
    """Verify exact published bytes, strictly load, then atomically publish an envelope."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    # Hash and load the same in-memory bytes, avoiding file-replacement races.
    raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in OFFICIAL_PIE_WEIGHTS:
        raise ValueError("Unknown checkpoint SHA-256; only verified official PIE NP K=1/K=20 weights are accepted")
    config = deepcopy(config.yaml_cfg if hasattr(config, "yaml_cfg") else config)
    config.pop("resume", None)
    config.pop("tuning", None)
    adapter = BiTraPAdapter(config)
    if adapter.pretrained_profile != "pie_np":
        raise ValueError("Use the pretrained PIE profile, not the generic BiTraP training profile")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a raw tensor state dictionary")
    adapter.build().load_state_dict(state, strict=True)
    trained_k, file_id = OFFICIAL_PIE_WEIGHTS[digest]
    adapter._pretrained_provenance = {
        "dataset": "PIE", "source_sha256": digest, "trained_num_samples": trained_k,
        "source_url": f"https://drive.google.com/uc?export=download&id={file_id}",
        "repository": adapter.source_repo, "license": adapter.license,
    }
    envelope = adapter.checkpoint()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bitrap-import-", suffix=".pth", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(envelope, stream)
        os.link(temporary, destination)  # Atomic, and never replaces an existing file.
    finally:
        Path(temporary).unlink(missing_ok=True)
    return envelope["pretrained_provenance"]
