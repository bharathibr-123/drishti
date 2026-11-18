"""Model loaders. Heavy imports (torch, ultralytics, open_clip) happen lazily, on first use."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import settings

if TYPE_CHECKING:
    from PIL import Image


class Encoder(Protocol):
    """Anything that maps images and texts into one shared, L2-normalised embedding space."""

    dim: int

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class ClipEncoder:
    def __init__(self, arch: str, pretrained: str) -> None:
        import open_clip
        import torch

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model: Any = model.eval()
        self.preprocess: Any = preprocess
        self.tokenizer: Any = open_clip.get_tokenizer(arch)
        self.dim = int(self.model.visual.output_dim)

    def _normalise(self, x: Any) -> np.ndarray:
        x = x / x.norm(dim=-1, keepdim=True)
        return np.asarray(x.float().cpu().numpy(), dtype=np.float32)

    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        with self._torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = [self.preprocess(im.convert("RGB")) for im in images[i : i + batch_size]]
                out.append(self._normalise(self.model.encode_image(self._torch.stack(batch))))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            return self._normalise(self.model.encode_text(self.tokenizer(texts)))


@functools.lru_cache(maxsize=1)
def get_encoder() -> ClipEncoder:
    return ClipEncoder(settings.clip_arch, settings.clip_pretrained)


def new_detector() -> Any:
    """A fresh YOLO instance. Tracker state lives in the instance, so use one per video stream."""
    from ultralytics import YOLO

    return YOLO(settings.detector_weights)
