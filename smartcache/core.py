"""
Core SmartCache logic - semantic caching for AI prompts.

Phases implemented:
- Phase 3: Cost-Aware LRU eviction
- Phase 4: Static Pinning of hot entries
- Phase 5: Input-Adaptive policy switching
- Bonus:   ML Predictor (Markov + Frequency) for proactive prefetching
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import hashlib
import json
import logging
import time
from dataclasses import dataclass

from .embeddings import EmbeddingEngine
from .storage import CacheStorage, CacheEntry, LMDBStorage
from .layer_predictor import LayerPredictor, InputAdaptivePolicy

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for SmartCache."""
    similarity_threshold: float = 0.92    # Default cosine similarity threshold
    cache_ttl: int = 3600 * 24 * 7       # 7 days
    enable_semantic: bool = True
    enable_exact_match: bool = True
    max_cache_size: int = 10000
    embedding_model: str = "all-MiniLM-L6-v2"

    # Phase 3: Cost-Aware LRU
    enable_cost_aware_eviction: bool = True

    # Phase 4: Static Pinning — keys to pin on startup / after first hit
    auto_pin_threshold: int = 10   # Pin entries accessed more than this many times

    # Phase 5: Input-Adaptive policy
    enable_adaptive_policy: bool = True

    # Bonus: ML Predictor
    enable_ml_predictor: bool = True
    prefetch_top_k: int = 3             # How many entries to prefetch proactively


class SmartCache:
    """
    Semantic caching layer for AI model calls.

    Enhancements over baseline:
    - Phase 3: Cost-Aware LRU eviction (evicts low-value entries first)
    - Phase 4: Static Pinning (hot entries stay in RAM forever)
    - Phase 5: Input-Adaptive policy (threshold adjusts to input length)
    - Bonus ML Predictor: Markov + frequency-based proactive prefetching

    Example:
        cache = SmartCache()
        client = OllamaClient(model="llama3", cache_storage=cache.storage)
        reply, meta = client.chat("What is Python?")
    """

    def __init__(
        self,
        storage: Optional[CacheStorage] = None,
        config: Optional[CacheConfig] = None
    ):
        self.config = config or CacheConfig()
        self.embedding_engine = EmbeddingEngine(self.config.embedding_model)
        self.storage = storage or LMDBStorage()

        # Phase 5 / Bonus: ML predictor
        self._predictor = LayerPredictor() if self.config.enable_ml_predictor else None

        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "total_requests": 0,
            "evictions": 0,
            "prefetch_hits": 0,   # Hits on proactively prefetched entries
        }
        self._start_time = time.time()

        # Phase 4: Track pinned key count
        self._pinned_keys: set = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        """Extract user prompt text from messages list."""
        user_content = " ".join(
            msg["content"] for msg in messages if msg.get("role") == "user"
        )
        return user_content[:2000]

    def _compute_hash(self, text: str, model: str, temperature: float) -> str:
        hash_input = f"{text}|{model}|{temperature:.2f}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _generate_cache_key(self, messages, model, temperature) -> str:
        return self._compute_hash(self._get_prompt_text(messages), model, temperature)

    def _compute_reload_cost(self, token_count: int) -> float:
        """
        Phase 3: Estimate the cost to regenerate this entry.
        Higher token count = more expensive to regenerate = keep longer.
        """
        return max(0.1, token_count / 1000.0)

    def _compute_layer_size(self, token_count: int, hit_count: int) -> float:
        """
        Phase 3: Entry 'size' in value terms.
        Frequently accessed entries with more tokens have higher value.
        """
        frequency_bonus = min(hit_count / 10.0, 5.0)  # Cap at 5x bonus
        return (token_count / 500.0) * (1.0 + frequency_bonus)

    def _maybe_auto_pin(self, key: str, entry: CacheEntry):
        """Phase 4: Auto-pin entries that exceed the hit threshold."""
        if (self.config.auto_pin_threshold > 0
                and entry.hit_count >= self.config.auto_pin_threshold
                and key not in self._pinned_keys):
            self.storage.pin(key)
            self._pinned_keys.add(key)
            logger.info(f"Auto-pinned hot entry: {key} (hits={entry.hit_count})")

    def _get_adaptive_threshold(self, prompt: str) -> Tuple[float, int, bool]:
        """
        Phase 5: Return (threshold, search_limit, prefer_exact) adapted to input.
        Falls back to config defaults if adaptive policy is disabled.
        """
        if not self.config.enable_adaptive_policy or self._predictor is None:
            return self.config.similarity_threshold, 5, True

        policy = InputAdaptivePolicy.get_policy(prompt)
        logger.debug(f"Adaptive policy: {policy['strategy_name']} "
                     f"(~{policy['token_estimate']} tokens, "
                     f"threshold={policy['similarity_threshold']})")
        return (
            policy["similarity_threshold"],
            policy["search_limit"],
            policy["prefer_exact"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_cache(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Check if response is cached.

        Returns:
            (cached_response, metadata) tuple if hit, None if miss.
        """
        self.stats["total_requests"] += 1
        prompt = self._get_prompt_text(messages)

        # Phase 5: Get adaptive threshold based on prompt length
        threshold, search_limit, prefer_exact = self._get_adaptive_threshold(prompt)

        # 1. Exact match (fast path)
        if self.config.enable_exact_match:
            exact_key = self._compute_hash(prompt, model, temperature)
            entry = self.storage.get(exact_key)
            if entry:
                self.stats["hits"] += 1
                self.stats["exact_hits"] += 1
                self.storage.increment_hit(exact_key)
                self._maybe_auto_pin(exact_key, entry)
                if self._predictor:
                    self._predictor.record_access(exact_key)
                logger.info(f"Cache HIT (exact): {exact_key}")
                meta = self._entry_to_meta(entry)
                meta["hit_type"] = "exact"
                meta["similarity"] = 1.0
                return entry.response, meta

            if prefer_exact:
                # For long prompts: skip semantic if exact not found
                # (high specificity — similar != correct)
                pass  # fall through to semantic anyway (configurable)

        # 2. Semantic similarity search with adaptive threshold
        if self.config.enable_semantic:
            query_embedding = self.embedding_engine.encode([prompt])[0]
            similar = self.storage.search_similar(
                query_embedding,
                threshold=threshold,
                limit=search_limit
            )

            if similar:
                best_key, best_entry, similarity = similar[0]
                self.stats["hits"] += 1
                self.stats["semantic_hits"] += 1
                self.storage.increment_hit(best_key)
                self._maybe_auto_pin(best_key, best_entry)
                if self._predictor:
                    self._predictor.record_access(best_key)
                logger.info(f"Cache HIT (semantic, {threshold:.2f}): {best_key} "
                            f"sim={similarity:.3f}")
                meta = self._entry_to_meta(best_entry)
                meta["hit_type"] = "semantic"
                meta["similarity"] = float(similarity)
                return best_entry.response, meta

        self.stats["misses"] += 1
        logger.info("Cache MISS")

        # Bonus: Trigger prefetch in background (record for next time)
        if self._predictor:
            self._trigger_prefetch()

        return None

    def cache_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        response: Any,
        token_count: int = 0
    ):
        """Store response in cache and enforce size limits."""
        prompt = self._get_prompt_text(messages)
        cache_key = self._compute_hash(prompt, model, temperature)

        existing = self.storage.get(cache_key)
        if existing:
            existing.hit_count += 1
            existing.reload_cost = self._compute_reload_cost(token_count)
            existing.layer_size = self._compute_layer_size(token_count, existing.hit_count)
            self.storage.set(cache_key, existing, existing.embedding)
            return

        embedding = self.embedding_engine.encode([prompt])[0]
        reload_cost = self._compute_reload_cost(token_count)
        layer_size = self._compute_layer_size(token_count, 0)

        entry = CacheEntry(
            prompt_hash=cache_key,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            hit_count=0,
            token_count=token_count,
            model=model,
            temperature=temperature,
            reload_cost=reload_cost,
            layer_size=layer_size,
        )
        self.storage.set(cache_key, entry, embedding)
        logger.info(f"Cached: {cache_key} (tokens={token_count}, cost={reload_cost:.2f})")

        # Phase 3: Enforce size limit using Cost-Aware LRU
        if self.config.enable_cost_aware_eviction:
            self.storage.evict_lru_cost_aware(self.config.max_cache_size)

    def pin(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.0):
        """
        Phase 4: Manually pin an entry so it's never evicted.
        Use this for known high-value entries (e.g., system-prompt responses).
        """
        prompt = self._get_prompt_text(messages)
        key = self._compute_hash(prompt, model, temperature)
        self.storage.pin(key)
        self._pinned_keys.add(key)

    def pin_by_key(self, key: str):
        """Phase 4: Pin by raw cache key."""
        self.storage.pin(key)
        self._pinned_keys.add(key)

    def _trigger_prefetch(self):
        """
        Bonus: Proactive prefetching based on ML predictor.
        In a real system this would warm the embedding index in background.
        Here we log candidates so the application layer can act on them.
        """
        if not self._predictor:
            return
        try:
            all_keys = [k for k, _ in self.storage.all_entries()]
            candidates = self._predictor.get_prefetch_candidates(
                all_keys, self.config.prefetch_top_k
            )
            if candidates:
                logger.debug(f"Prefetch candidates: {candidates}")
        except AttributeError:
            pass  # Storage backend may not support all_entries

    def get_prefetch_candidates(self) -> List[str]:
        """Bonus: Return keys the predictor thinks should be prefetched."""
        if not self._predictor:
            return []
        try:
            all_keys = [k for k, _ in self.storage.all_entries()]
            return self._predictor.get_prefetch_candidates(all_keys, self.config.prefetch_top_k)
        except AttributeError:
            return []

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total_requests"]
        hits = self.stats["hits"]
        hit_rate = hits / total if total > 0 else 0.0

        stats = {
            **self.stats,
            "hit_rate": hit_rate,
            "uptime": time.time() - self._start_time,
            "pinned_count": len(self._pinned_keys),
        }

        # Add predictor stats if enabled
        if self._predictor:
            stats["predictor"] = self._predictor.get_stats()

        # Token savings
        try:
            all_entries = self.storage.all_entries()
            stats["total_saved_tokens"] = sum(
                e.token_count * max(0, e.hit_count)
                for _, e in all_entries
            )
            stats["cache_entry_count"] = len(all_entries)
        except AttributeError:
            stats["total_saved_tokens"] = 0

        return stats

    def _entry_to_meta(self, entry: CacheEntry) -> Dict[str, Any]:
        return {
            "cached": True,
            "hit_count": entry.hit_count,
            "created_at": entry.created_at,
            "model": entry.model,
            "token_count": entry.token_count,
            "cache_age": time.time() - entry.created_at,
            "pinned": entry.pinned,
            "reload_cost": entry.reload_cost,
            "layer_size": entry.layer_size,
        }

    def _get_all_entries(self):
        try:
            return [e for _, e in self.storage.all_entries()]
        except AttributeError:
            return []

    def clear(self):
        """Clear all non-pinned cache entries."""
        try:
            for key, entry in self.storage.all_entries():
                if not entry.pinned:
                    self.storage._delete(key)
            logger.info("Cache cleared (pinned entries preserved)")
        except AttributeError:
            logger.warning("clear() not fully supported by this storage backend")

    def preload(
        self,
        prompt_response_pairs: List[Tuple[str, Any, str, float, int]]
    ):
        """
        Pre-populate cache with known Q&A pairs.

        Args:
            prompt_response_pairs: List of (prompt_text, response, model, temperature, token_count)
        """
        for prompt, response, model, temperature, tokens in prompt_response_pairs:
            messages = [{"role": "user", "content": prompt}]
            self.cache_response(messages, model, temperature, response, tokens)
        logger.info(f"Preloaded {len(prompt_response_pairs)} entries")
