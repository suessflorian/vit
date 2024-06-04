import torch
import os
import io
import upload
from dataclasses import dataclass
from typing import Tuple
from torchvision.models import VisionTransformer
from google.cloud import storage

CACHE = "./checkpoint"
PREFIX = "checkpoint"
BUCKET_NAME = "florians_results"

@dataclass
class Metadata:
    name: str
    epoch: int
    accuracy: float
    loss: float

def cache(model: VisionTransformer, metadata: Metadata, gcs: bool = False):
    # TODO: short-term to save model on CPU to ensure compatible loading between devices.
    device = next(model.parameters()).device
    model.to("cpu")
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    model.to(device)

    # NOTE: save the epoch model for historical winding if nessecary.
    path = f"{CACHE}/{metadata.name}-{metadata.epoch}-checkpoint.pth"
    gcs_path = f"{PREFIX}/{metadata.name}-{metadata.epoch}-checkpoint.pth"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(checkpoint, path)

    if gcs:
        upload.gcs(path, gcs_path)

    # NOTE: by default override the latest version of the model by default
    path = f"{CACHE}/{metadata.name}-checkpoint.pth"
    gcs_path = f"{PREFIX}/{metadata.name}-checkpoint.pth"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(checkpoint, path)

    if gcs:
        upload.gcs(path, gcs_path)


# load retrieves the latest model
def load(model: VisionTransformer, name: str, gcs: bool = False) -> Tuple[bool, VisionTransformer, Metadata]:
    path = f"{CACHE}/{name}-checkpoint.pth"
    gcs_path = f"{PREFIX}/{name}-checkpoint.pth"

    if gcs:
        print("-> loading from GCS")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            print("-> nothing found in GCS")
        else:
            checkpoint_data = blob.download_as_bytes()
            checkpoint = torch.load(io.BytesIO(checkpoint_data))
            model.load_state_dict(checkpoint["state_dict"])
            metadata = checkpoint["metadata"]
            return True, model, metadata

    print("-> loading from disk")
    if not os.path.exists(path):
        print("-> nothing found on disk")
        return False, model, Metadata(name, 0, 0, float("inf"))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = checkpoint["metadata"]
    return True, model, metadata
