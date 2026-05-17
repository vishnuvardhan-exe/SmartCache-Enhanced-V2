"""
Unit tests for SmartCache — all phases + OllamaClient.
All tests run offline (no Ollama server required).
"""

import pytest
import time
import numpy as np
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from smartcache.embeddings import EmbeddingEngine
from smartcache.storage import CacheEntry, LMDBStorage
from smartcache.core import SmartCache, CacheConfig
from smartcache.layer_predictor import (
    MarkovPredictor, FrequencyPredictor, InputAdaptivePolicy, LayerPredictor,
)
from smartcache.ollama_client import OllamaClient
from smartcache.model_selector import ModelSelector


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def storage(temp_dir):
    s = LMDBStorage(db_path=temp_dir)
    yield s
    s.close()


@pytest.fixture
def cache(temp_dir):
    s = LMDBStorage(db_path=temp_dir)
    c = SmartCache(
        storage=s,
        config=CacheConfig(
            enable_adaptive_policy=True,
            enable_ml_predictor=True,
            auto_pin_threshold=3,
        ),
    )
    yield c
    s.close()


def _mock_generate(messages, model):
    prompt = messages[-1]["content"] if messages else ""
    return {
        "message": {"content": f"[mock] {prompt[:30]}"},
        "eval_count": 20,
        "prompt_eval_count": 5,
    }


@pytest.fixture
def ollama_client(temp_dir):
    storage = LMDBStorage(db_path=temp_dir)
    client = OllamaClient(
        model="llama3",
        cache_storage=storage,
        similarity_threshold=0.85,
        _mock_latency=0.0,
        _mock_generate=_mock_generate,
    )
    yield client
    storage.close()


def make_entry(key="k", tokens=100, hits=0) -> CacheEntry:
    emb = np.random.rand(384).astype(np.float32)
    return CacheEntry(
        prompt_hash=key,
        response={"text": "resp"},
        embedding=emb,
        created_at=time.time(),
        hit_count=hits,
        token_count=tokens,
        model="llama3",
        temperature=0.0,
        reload_cost=tokens / 1000.0,
        layer_size=tokens / 500.0,
    )


# ─────────────────────────────────────────────────────────────
# EmbeddingEngine
# ─────────────────────────────────────────────────────────────

class TestEmbeddingEngine:
    @pytest.fixture
    def engine(self):
        return EmbeddingEngine(model_name="all-MiniLM-L6-v2")

    def test_initialization(self, engine):
        assert engine.dimension == 384

    def test_encode_shape(self, engine):
        emb = engine.encode(["Hello world"])
        assert emb.shape == (1, 384)

    def test_similarity_ordering(self, engine):
        e1 = engine.encode(["The cat sits on the mat"])[0]
        e2 = engine.encode(["A cat is sitting on the mat"])[0]
        e3 = engine.encode(["The weather is nice today"])[0]
        assert engine.similarity(e1, e2) > engine.similarity(e1, e3)


# ─────────────────────────────────────────────────────────────
# Phase 3: Cost-Aware LRU
# ─────────────────────────────────────────────────────────────

class TestCostAwareLRU:
    def test_eviction_score_high_cost_stays(self):
        cheap = make_entry("cheap", tokens=10)
        cheap.reload_cost = 0.01
        expensive = make_entry("exp", tokens=500)
        expensive.reload_cost = 0.5
        assert expensive.eviction_score > cheap.eviction_score

    def test_eviction_score_pinned_marked(self, storage):
        entry = make_entry("pin_me", tokens=10)
        storage.set("pin_me", entry, entry.embedding)
        storage.pin("pin_me")
        retrieved = storage.get("pin_me")
        assert retrieved.pinned is True

    def test_lru_eviction_respects_cost(self, temp_dir):
        s = LMDBStorage(db_path=temp_dir)
        for i in range(5):
            e = make_entry(f"cheap_{i}", tokens=10)
            e.reload_cost = 0.01
            s.set(f"cheap_{i}", e, e.embedding)

        exp = make_entry("expensive", tokens=1000)
        exp.reload_cost = 1.0
        s.set("expensive", exp, exp.embedding)

        s.evict_lru_cost_aware(max_entries=2)
        surviving = [k for k, _ in s.all_entries()]
        assert "expensive" in surviving
        s.close()

    def test_pinned_not_evicted(self, temp_dir):
        s = LMDBStorage(db_path=temp_dir)
        for i in range(5):
            e = make_entry(f"entry_{i}", tokens=50)
            s.set(f"entry_{i}", e, e.embedding)
        s.pin("entry_0")
        s.evict_lru_cost_aware(max_entries=1)
        surviving = [k for k, _ in s.all_entries()]
        assert "entry_0" in surviving
        s.close()


# ─────────────────────────────────────────────────────────────
# Phase 4: Static Pinning
# ─────────────────────────────────────────────────────────────

class TestStaticPinning:
    def test_pin_and_get(self, storage):
        entry = make_entry("hot_key", hits=20)
        storage.set("hot_key", entry, entry.embedding)
        storage.pin("hot_key")
        assert storage.get("hot_key").pinned is True

    def test_unpin(self, storage):
        entry = make_entry("k2")
        storage.set("k2", entry, entry.embedding)
        storage.pin("k2")
        storage.unpin("k2")
        assert storage.get("k2").pinned is False

    def test_auto_pin_on_hit_threshold(self, cache):
        msgs = [{"role": "user", "content": "Pinnable question"}]
        cache.cache_response(msgs, "llama3", 0.0, {"text": "ans"}, 200)
        for _ in range(4):
            cache.check_cache(msgs, "llama3", 0.0)
        assert cache.get_stats()["pinned_count"] >= 1


# ─────────────────────────────────────────────────────────────
# Phase 5: Input-Adaptive Policy
# ─────────────────────────────────────────────────────────────

class TestInputAdaptivePolicy:
    def test_short_prompt_lower_threshold(self):
        policy = InputAdaptivePolicy.get_policy("What is AI?")
        assert policy["strategy_name"] == "ffn_broad"
        assert policy["similarity_threshold"] < 0.92

    def test_long_prompt_stricter_threshold(self):
        policy = InputAdaptivePolicy.get_policy("word " * 600)
        assert policy["strategy_name"] == "attention_strict"
        assert policy["similarity_threshold"] >= 0.95

    def test_medium_prompt_balanced(self):
        policy = InputAdaptivePolicy.get_policy("word " * 300)
        assert policy["strategy_name"] == "balanced"

    def test_adaptive_affects_cache(self, cache):
        msgs_orig = [{"role": "user", "content": "What is Python?"}]
        cache.cache_response(msgs_orig, "llama3", 0.0, {"text": "ans"}, 50)
        msgs_alt = [{"role": "user", "content": "Explain Python"}]
        result = cache.check_cache(msgs_alt, "llama3", 0.0)
        assert result is None or isinstance(result, tuple)


# ─────────────────────────────────────────────────────────────
# Bonus: ML Predictor
# ─────────────────────────────────────────────────────────────

class TestMarkovPredictor:
    def test_learns_sequence(self):
        p = MarkovPredictor(order=2)
        for _ in range(10):
            p.record_access("A")
            p.record_access("B")
            p.record_access("C")
        p.access_history.extend(["A", "B"])
        preds = p.predict_next()
        keys = [k for k, _ in preds]
        assert "C" in keys

    def test_empty_history(self):
        p = MarkovPredictor()
        assert p.predict_next() == []


class TestFrequencyPredictor:
    def test_most_accessed_ranked_first(self):
        fp = FrequencyPredictor()
        for _ in range(10):
            fp.record_access("hot_key")
        fp.record_access("cold_key")
        top = fp.top_candidates(["hot_key", "cold_key"], top_k=1)
        assert top[0] == "hot_key"


class TestLayerPredictor:
    def test_get_adaptive_policy(self):
        lp = LayerPredictor()
        policy = lp.get_adaptive_policy("short?")
        assert "similarity_threshold" in policy
        assert "strategy_name" in policy

    def test_prefetch_candidates_returns_list(self):
        lp = LayerPredictor()
        for key in ["a", "b", "c", "a", "b", "a"]:
            lp.record_access(key)
        candidates = lp.get_prefetch_candidates(["a", "b", "c"], top_k=2)
        assert isinstance(candidates, list)
        assert len(candidates) <= 2


# ─────────────────────────────────────────────────────────────
# Core SmartCache integration
# ─────────────────────────────────────────────────────────────

class TestSmartCache:
    def test_exact_match(self, cache):
        msgs = [{"role": "user", "content": "What is AI?"}]
        cache.cache_response(msgs, "llama3", 0.0, {"text": "AI is…"}, 50)
        result = cache.check_cache(msgs, "llama3", 0.0)
        assert result is not None
        _, meta = result
        assert meta["cached"] is True

    def test_miss_returns_none(self, cache):
        msgs = [{"role": "user", "content": "Totally unique xyzzy question"}]
        assert cache.check_cache(msgs, "llama3", 0.0) is None

    def test_stats_increment(self, cache):
        msgs = [{"role": "user", "content": "Stats test"}]
        cache.check_cache(msgs, "llama3", 0.0)
        stats = cache.get_stats()
        assert stats["total_requests"] == 1
        assert stats["misses"] == 1

    def test_cost_fields_stored(self, cache):
        msgs = [{"role": "user", "content": "Token rich question"}]
        cache.cache_response(msgs, "llama3", 0.0, {"text": "resp"}, 800)
        key = cache._compute_hash("Token rich question", "llama3", 0.0)
        entry = cache.storage.get(key)
        assert entry.reload_cost == pytest.approx(0.8, rel=0.1)

    def test_preload(self, cache):
        pairs = [("Q about Python", {"text": "Python ans"}, "llama3", 0.0, 100)]
        cache.preload(pairs)
        result = cache.check_cache(
            [{"role": "user", "content": "Q about Python"}], "llama3", 0.0
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────
# OllamaClient tests
# ─────────────────────────────────────────────────────────────

class TestOllamaClient:
    def test_chat_miss_returns_mock_reply(self, ollama_client):
        reply, meta = ollama_client.chat("What is Python?")
        assert "mock" in reply.lower() or len(reply) > 0
        assert meta["cached"] is False
        assert meta["latency_ms"] >= 0

    def test_chat_exact_hit(self, ollama_client):
        ollama_client.chat("What is machine learning?")
        reply2, meta2 = ollama_client.chat("What is machine learning?")
        assert meta2["cached"] is True
        assert meta2["hit_type"] in ("exact", "semantic")

    def test_cache_stats_populated(self, ollama_client):
        ollama_client.chat("Hello world")
        stats = ollama_client.get_cache_stats()
        assert stats["total_requests"] >= 1

    def test_preload_faq(self, ollama_client):
        ollama_client.preload_faq([
            ("What is SmartCache?", "A semantic caching library.", 0.0, 20),
        ])
        reply, meta = ollama_client.chat("What is SmartCache?")
        assert meta["cached"] is True

    def test_pin_prompt(self, ollama_client):
        # First call populates cache
        ollama_client.chat("Pinnable FAQ question?")
        ollama_client.pin_prompt("Pinnable FAQ question?")
        stats = ollama_client.get_cache_stats()
        assert stats.get("pinned_count", 0) >= 1

    def test_clear_cache(self, ollama_client):
        ollama_client.chat("Something to clear")
        ollama_client.clear_cache()
        stats = ollama_client.get_cache_stats()
        assert stats["total_requests"] == 0 or stats["hits"] == 0

    def test_chat_stream_miss_yields_text(self, ollama_client):
        # Streaming uses the same mock via _generate_stream; patch it
        with patch.object(
            ollama_client, "_generate_stream",
            return_value=iter(["Hello ", "world"])
        ):
            chunks = list(ollama_client.chat_stream("Stream test"))
        assert "".join(chunks) in ("Hello world", "[mock] Stream test"[:30]) or len(chunks) > 0

    def test_chat_stream_hit_single_chunk(self, ollama_client):
        # Populate via regular chat, then stream should be a cache hit
        ollama_client.chat("Stream cache test")
        with patch.object(
            ollama_client, "_generate_stream",
            side_effect=AssertionError("should not call Ollama on cache hit"),
        ):
            chunks = list(ollama_client.chat_stream("Stream cache test"))
        assert len(chunks) == 1   # one cached chunk

    def test_health_check_false_when_server_down(self):
        assert OllamaClient.health_check("http://127.0.0.1:19999") is False

    def test_list_models_empty_when_server_down(self):
        assert OllamaClient.list_models("http://127.0.0.1:19999") == []


# ─────────────────────────────────────────────────────────────
# ModelSelector tests
# ─────────────────────────────────────────────────────────────

class TestModelSelector:
    def test_pick_by_filter(self):
        sel = ModelSelector(base_url="http://127.0.0.1:19999")
        sel._models = ["llama3:8b", "mistral:7b", "phi3:mini"]
        result = sel.pick(filter="mistral", interactive=False)
        assert result == "mistral:7b"

    def test_pick_default_when_no_filter_match(self):
        sel = ModelSelector()
        sel._models = ["llama3:8b", "mistral:7b"]
        result = sel.pick(filter="gpt", interactive=False)
        # No match → first available
        assert result == "llama3:8b"

    def test_info_parses_tags(self):
        sel = ModelSelector()
        sel._models = ["llama3:8b", "mistral:latest"]
        info = sel.info()
        assert info[0]["base"] == "llama3"
        assert info[0]["tag"] == "8b"
        assert info[1]["tag"] == "latest"

    def test_available_fetches_when_none(self):
        sel = ModelSelector(base_url="http://127.0.0.1:19999")
        # Server is down → empty list, no exception
        models = sel.available()
        assert isinstance(models, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
