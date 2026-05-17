"""
SmartCache: Semantic caching layer for local Ollama inference.

Phases:
- Phase 3: Cost-Aware LRU eviction
- Phase 4: Static Pinning
- Phase 5: Input-Adaptive Policy
- Bonus:   ML Predictor (Markov + Frequency)
"""

from .core import SmartCache, CacheConfig
from .ollama_client import OllamaClient
from .model_selector import ModelSelector
from .embeddings import EmbeddingEngine
from .storage import CacheStorage, LMDBStorage, RedisStorage
from .layer_predictor import LayerPredictor, InputAdaptivePolicy

__version__ = "2.0.0"
__author__ = "SmartCache Team"

__all__ = [
    "SmartCache",
    "CacheConfig",
    "OllamaClient",
    "ModelSelector",
    "EmbeddingEngine",
    "CacheStorage",
    "LMDBStorage",
    "RedisStorage",
    "LayerPredictor",
    "InputAdaptivePolicy",
]
