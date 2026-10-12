"""Cross-encoder reranker (notebook ``CrossEncoderWrapper``)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderWrapper:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", model_name)
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype="float32")
        return np.asarray(self.model.predict(pairs), dtype="float32")
