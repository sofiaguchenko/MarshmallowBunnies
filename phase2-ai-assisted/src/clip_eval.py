from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

PROMPT_LADDER: Dict[str, List[str]] = {
    "L1_minimal": [
        "{genre}",
    ],
    "L2_neutral": [
        "A book cover of a {genre} book.",
    ],
    "L3_stylistic_mood": [
        "A visually rich book cover with a strong {genre} mood.",
    ],
    "L4_prompt_ensemble": [
        "A book cover of a {genre} book.",
        "The front cover of a {genre} novel or book.",
        "A published book cover in the {genre} genre.",
        "Book cover art and typography for a {genre} book.",
    ],
}


def get_device() -> str:
    """Select the best available compute device (CUDA, Apple Silicon MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> Tuple[CLIPModel, CLIPProcessor, str]:
    """Load CLIP model and processor onto the target device in eval mode."""
    target_device = device or get_device()
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(target_device)
    model.eval()
    return model, processor, target_device


@torch.no_grad()
def build_genre_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    genres: List[str],
    templates: List[str],
    device: str,
) -> torch.Tensor:
    """Encode and L2-normalize class text prototypes across one or more prompt templates."""
    class_embeddings: List[torch.Tensor] = []

    for genre in genres:
        prompts = [tmpl.format(genre=genre) for tmpl in templates]
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Average across templates (no-op for single-template levels) and re-normalize
        prototype = text_features.mean(dim=0)
        prototype = F.normalize(prototype, p=2, dim=-1)
        class_embeddings.append(prototype)

    return torch.stack(class_embeddings, dim=0)  # Shape: (num_genres, embed_dim)


@torch.no_grad()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: List[Path],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute L2-normalized CLIP image embeddings in batches."""
    all_embeddings: List[torch.Tensor] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt").to(device)
        img_features = model.get_image_features(**inputs)
        img_features = F.normalize(img_features, p=2, dim=-1)
        all_embeddings.append(img_features.cpu())

    return torch.cat(all_embeddings, dim=0)  # Shape: (num_images, embed_dim)


def score_embeddings(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    genres: List[str],
    logit_scale: float,
    prefix: str,
) -> pd.DataFrame:
    """Compute top-1/top-2 predictions, similarity margins, and calibrated probabilities."""
    sims = image_embeddings @ text_embeddings.cpu().T  # (N, C)
    probs = torch.softmax(sims * logit_scale, dim=-1)

    top2_sims, top2_idx = torch.topk(sims, k=2, dim=-1)
    top1_probs = probs.gather(1, top2_idx[:, :1]).squeeze(1)

    sims_np = top2_sims.numpy()
    idx_np = top2_idx.numpy()
    probs_np = top1_probs.numpy()

    return pd.DataFrame(
        {
            f"{prefix}_pred": [genres[i] for i in idx_np[:, 0]],
            f"{prefix}_top2": [genres[i] for i in idx_np[:, 1]],
            f"{prefix}_top1_sim": sims_np[:, 0],
            f"{prefix}_top2_sim": sims_np[:, 1],
            f"{prefix}_margin": sims_np[:, 0] - sims_np[:, 1],
            f"{prefix}_prob": probs_np,
        }
    )
