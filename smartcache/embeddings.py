"""
Embedding engine for semantic similarity.
Uses sentence-transformers for local, fast embeddings.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates embeddings for semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: SentenceTransformer model name.
                all-MiniLM-L6-v2 is fast and small (384 dims)
                all-mpnet-base-v2 is higher quality (768 dims)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        return embeddings.astype(np.float32)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize and compute dot product
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
