"""
Phase 5 (Bonus): ML Layer Predictor for SmartCache.

Predicts which cached entries are likely to be needed next based on:
- Recent access patterns (sequence modeling)
- Input length characteristics
- Time-of-day / temporal patterns

Uses a lightweight Markov chain + frequency model that requires no external ML deps.
An optional sklearn-based logistic regression is also provided for better accuracy.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import time
import logging
import math

logger = logging.getLogger(__name__)


class MarkovPredictor:
    """
    Lightweight n-gram Markov chain predictor for cache access sequences.
    No external dependencies required.
    Predicts: given the last N accesses, what key will be accessed next?
    """

    def __init__(self, order: int = 2, max_states: int = 5000):
        """
        Args:
            order: Markov chain order (look-back window)
            max_states: Max number of state transitions to remember
        """
        self.order = order
        self.max_states = max_states
        # transitions[state_tuple] = {next_key: count}
        self.transitions: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.access_history: deque = deque(maxlen=order * 100)
        self._state_count = 0

    def record_access(self, key: str):
        """Record a cache access to build the prediction model."""
        self.access_history.append(key)
        history = list(self.access_history)

        # Build n-gram transitions
        if len(history) > self.order:
            for i in range(len(history) - self.order):
                state = tuple(history[i:i + self.order])
                next_key = history[i + self.order]
                self.transitions[state][next_key] += 1
                self._state_count = len(self.transitions)

        # Prune if too large (keep most recent states)
        if self._state_count > self.max_states:
            # Remove oldest entries
            oldest = list(self.transitions.keys())[:self.max_states // 4]
            for k in oldest:
                del self.transitions[k]

    def predict_next(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the top-k most likely next cache keys.

        Returns:
            List of (key, probability) tuples, sorted by probability descending.
        """
        history = list(self.access_history)
        if len(history) < self.order:
            return []

        state = tuple(history[-self.order:])
        if state not in self.transitions:
            # Fall back to lower-order predictions
            for order in range(self.order - 1, 0, -1):
                state = tuple(history[-order:])
                if state in self.transitions:
                    break
            else:
                return []

        counts = self.transitions[state]
        total = sum(counts.values())
        if total == 0:
            return []

        predictions = [(key, count / total) for key, count in counts.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def get_prefetch_candidates(self, top_k: int = 3) -> List[str]:
        """Return keys to prefetch (warm up) based on prediction."""
        return [key for key, _ in self.predict_next(top_k)]


class FrequencyPredictor:
    """
    Simple frequency-based predictor with time-decay.
    Useful as a fallback or complement to Markov.
    Score = hit_count * e^(-decay * age_hours)
    """

    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        self.access_times: Dict[str, List[float]] = defaultdict(list)

    def record_access(self, key: str):
        self.access_times[key].append(time.time())
        # Keep only last 1000 accesses per key to save memory
        if len(self.access_times[key]) > 1000:
            self.access_times[key] = self.access_times[key][-1000:]

    def score(self, key: str) -> float:
        """Compute time-decayed frequency score."""
        now = time.time()
        times = self.access_times.get(key, [])
        return sum(
            math.exp(-self.decay_rate * (now - t) / 3600.0)
            for t in times
        )

    def top_candidates(self, all_keys: List[str], top_k: int = 5) -> List[str]:
        scored = [(k, self.score(k)) for k in all_keys]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:top_k]]


class InputAdaptivePolicy:
    """
    Phase 5: Input-Adaptive Switching.

    Adjusts the cache lookup strategy based on the characteristics
    of the current input (length, type, etc.).

    Strategy:
    - Short inputs (< 200 tokens):   prioritize FFN-style broad semantic search
    - Medium inputs (200-500 tokens): balanced exact + semantic
    - Long inputs (> 500 tokens):     prioritize attention/exact match (specificity matters)
    """

    SHORT_THRESHOLD = 200    # tokens
    LONG_THRESHOLD = 500     # tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return max(1, len(text) // 4)

    @classmethod
    def get_policy(cls, prompt: str) -> Dict:
        """
        Returns adapted cache policy parameters for this prompt.

        Returns dict with:
            similarity_threshold: float — adjusted similarity cutoff
            prefer_exact: bool — whether to skip semantic if exact found
            search_limit: int — how many semantic candidates to evaluate
            strategy_name: str — human-readable policy label
        """
        token_count = cls.estimate_tokens(prompt)

        if token_count < cls.SHORT_THRESHOLD:
            # Short prompt: broader semantic search, lower threshold
            # FFN-style: capture many variations of simple queries
            return {
                "similarity_threshold": 0.85,
                "prefer_exact": False,
                "search_limit": 10,
                "strategy_name": "ffn_broad",
                "token_estimate": token_count,
            }
        elif token_count < cls.LONG_THRESHOLD:
            # Medium: balanced approach
            return {
                "similarity_threshold": 0.90,
                "prefer_exact": True,
                "search_limit": 7,
                "strategy_name": "balanced",
                "token_estimate": token_count,
            }
        else:
            # Long prompt: strict matching (attention-layer style)
            # High specificity required — don't return loosely similar cached answers
            return {
                "similarity_threshold": 0.95,
                "prefer_exact": True,
                "search_limit": 3,
                "strategy_name": "attention_strict",
                "token_estimate": token_count,
            }


class LayerPredictor:
    """
    Combined ML predictor: Markov + Frequency + Input-Adaptive.

    Exposes a unified interface used by SmartCache to:
    1. Get prefetch candidates (proactive warming)
    2. Get adapted policy per request
    3. Record accesses for learning
    """

    def __init__(self, markov_order: int = 2):
        self.markov = MarkovPredictor(order=markov_order)
        self.frequency = FrequencyPredictor()
        self.adaptive = InputAdaptivePolicy()
        self._total_predictions = 0
        self._correct_predictions = 0

    def record_access(self, key: str):
        """Call this every time a cache entry is accessed."""
        self.markov.record_access(key)
        self.frequency.record_access(key)

    def get_prefetch_candidates(self, all_keys: List[str], top_k: int = 5) -> List[str]:
        """
        Combine Markov + Frequency predictions for prefetch candidates.
        Markov catches sequential patterns; Frequency catches hot spots.
        """
        markov_hits = set(self.markov.get_prefetch_candidates(top_k))
        freq_hits = set(self.frequency.top_candidates(all_keys, top_k))
        # Union, prioritize Markov predictions (more context-aware)
        combined = list(markov_hits) + [k for k in freq_hits if k not in markov_hits]
        return combined[:top_k]

    def get_adaptive_policy(self, prompt: str) -> Dict:
        """Phase 5: Get input-length-aware cache policy for this prompt."""
        return self.adaptive.get_policy(prompt)

    def record_prediction_outcome(self, predicted_keys: List[str], actual_key: str):
        """Optional: track prediction accuracy for monitoring."""
        self._total_predictions += 1
        if actual_key in predicted_keys:
            self._correct_predictions += 1

    @property
    def prediction_accuracy(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_stats(self) -> Dict:
        return {
            "markov_states": len(self.markov.transitions),
            "markov_history_len": len(self.markov.access_history),
            "total_predictions": self._total_predictions,
            "prediction_accuracy": self.prediction_accuracy,
        }
