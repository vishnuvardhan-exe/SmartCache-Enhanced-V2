#!/usr/bin/env python3
"""
SmartCache Example — demonstrating OllamaClient with semantic caching.

Run:  python example.py

Ollama does NOT need to be running — the example uses a mock so it
works entirely offline.
"""

import time
from unittest.mock import patch, MagicMock
from smartcache import OllamaClient, SmartCache, CacheConfig
from smartcache.storage import LMDBStorage
import tempfile, shutil


# ──────────────────────────────────────────────
# 1. Basic caching demo (offline-safe mock)
# ──────────────────────────────────────────────

def demo_basic_caching():
    print("=" * 60)
    print("SmartCache Demo: OllamaClient with Semantic Caching")
    print("=" * 60)

    td = tempfile.mkdtemp()
    try:
        storage = LMDBStorage(db_path=td)
        client = OllamaClient(
            model="llama3",
            cache_storage=storage,
            similarity_threshold=0.85,
            _mock_latency=0.05,   # simulated inference: 50 ms
            _mock_generate=lambda msgs, model: {
                "message": {"content": f"[mock] answer for: {msgs[-1]['content']}"},
                "eval_count": 30,
                "prompt_eval_count": 10,
            },
        )
        print("✓ OllamaClient initialised (mock mode)\n")

        test_prompts = [
            "What is Python?",
            "Explain Python programming",     # semantic hit expected
            "What's Python used for?",         # near-hit
            "How do I learn machine learning?",
            "Best way to learn ML",            # semantic hit expected
            "Tell me about quantum computing",
            "What is quantum computing?",      # semantic hit expected
        ]

        print("[1] Populating cache with initial prompts...")
        seed_prompts = [test_prompts[0], test_prompts[3], test_prompts[5]]
        for p in seed_prompts:
            reply, meta = client.chat(p)
            print(f"  SEED  | {p[:45]:45} | {meta['latency_ms']:.1f}ms")

        print("\n[2] Testing cache lookup on all prompts...")
        total_saved_ms = 0.0
        for p in test_prompts:
            reply, meta = client.chat(p)
            hit_label = "HIT " if meta["cached"] else "MISS"
            saved = meta.get("time_saved_ms", 0.0)
            total_saved_ms += saved
            print(
                f"  {hit_label} | {p[:45]:45} | "
                f"{meta['latency_ms']:6.1f}ms | "
                f"saved ~{saved:.0f}ms"
            )

        print("\n[3] Cache statistics")
        stats = client.get_cache_stats()
        print(f"  Total requests : {stats['total_requests']}")
        print(f"  Hits           : {stats['hits']}  "
              f"(exact: {stats.get('exact_hits', 0)}, "
              f"semantic: {stats.get('semantic_hits', 0)})")
        print(f"  Misses         : {stats['misses']}")
        print(f"  Hit rate       : {stats['hit_rate']*100:.1f}%")
        print(f"  Est. time saved: {total_saved_ms:.0f} ms total")

    finally:
        storage.close()
        shutil.rmtree(td)

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start Ollama:         ollama serve")
    print("  2. Pull a model:         ollama pull llama3")
    print("  3. Run the Streamlit UI: streamlit run demo/app.py")
    print("  4. Run full benchmark:   python benchmark.py")


# ──────────────────────────────────────────────
# 2. Live Ollama demo (requires Ollama running)
# ──────────────────────────────────────────────

def demo_live_ollama():
    """Demo with a real Ollama server.  Only runs if Ollama is reachable."""
    from smartcache import OllamaClient
    import tempfile, shutil

    if not OllamaClient.health_check():
        print("\n⚠️  Ollama not running — skipping live demo.")
        print("   Start it with:  ollama serve")
        return

    models = OllamaClient.list_models()
    if not models:
        print("\n⚠️  No models pulled.  Run:  ollama pull llama3")
        return

    model = models[0]
    print(f"\n[Live] Using model: {model}")

    td = tempfile.mkdtemp()
    try:
        client = OllamaClient(model=model, cache_storage=LMDBStorage(db_path=td))

        queries = [
            "What is machine learning?",
            "Explain machine learning",
            "Define machine learning",
        ]

        for q in queries:
            t0 = time.time()
            reply, meta = client.chat(q, temperature=0.0)
            elapsed = time.time() - t0
            status = "HIT " if meta["cached"] else "MISS"
            print(f"  {status} | {q[:40]:40} | {elapsed:.2f}s | {reply[:50]}...")

        stats = client.get_cache_stats()
        print(f"\n  Hit rate: {stats['hit_rate']*100:.1f}%")
    finally:
        shutil.rmtree(td)


if __name__ == "__main__":
    demo_basic_caching()
    demo_live_ollama()
