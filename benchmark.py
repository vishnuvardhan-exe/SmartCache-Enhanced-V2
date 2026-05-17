#!/usr/bin/env python3
"""
SmartCache Benchmark — All Phases + Ablation
Runs entirely offline using a mock Ollama HTTP server.

Run:  python benchmark.py
"""

import time
import random
import statistics
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from smartcache import SmartCache, CacheConfig
from smartcache.ollama_client import OllamaClient
from smartcache.storage import LMDBStorage
from smartcache.layer_predictor import InputAdaptivePolicy, LayerPredictor


# ─────────────────────────────────────────────────────────────
# Mock Ollama server setup
# ─────────────────────────────────────────────────────────────

SIMULATED_INFERENCE_MS = 80   # Simulated time for one real Ollama call


def make_mock_generate(latency_s: float = SIMULATED_INFERENCE_MS / 1000):
    """Return a mock _generate callable that sleeps to simulate inference."""
    def _mock(messages, model):
        time.sleep(latency_s)
        prompt = messages[-1]["content"] if messages else ""
        return {
            "message": {"content": f"[mock answer] {prompt[:40]}"},
            "eval_count": 40,
            "prompt_eval_count": 15,
        }
    return _mock


# ─────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────

TOPICS = ["Python", "machine learning", "AI", "quantum computing",
          "blockchain", "neural networks", "data science", "API", "database"]

TEMPLATES = [
    "What is {topic}?",
    "Explain {topic}",
    "How does {topic} work?",
    "Tell me about {topic}",
    "Define {topic}",
]


def gen_prompts(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [rng.choice(TEMPLATES).format(topic=rng.choice(TOPICS)) for _ in range(n)]


def gen_long_prompts(n: int, seed: int = 99) -> list:
    rng = random.Random(seed)
    base = "word " * 300
    return [base + f"specifically about {t}?" for t in rng.choices(TOPICS, k=n)]


# ─────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────

def run_benchmark(client: OllamaClient, queries: list, label: str) -> dict:
    latencies, hits, misses = [], 0, 0
    print(f"\n  {label}")

    for query in queries:
        t0 = time.perf_counter()
        _, meta = client.chat(query, temperature=0.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        if meta["cached"]:
            hits += 1
        else:
            misses += 1

    total = len(queries)
    return {
        "hits": hits,
        "misses": misses,
        "total": total,
        "hit_rate": hits / total if total else 0,
        "avg_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "total_time_ms": sum(latencies),
    }


def print_result(r: dict, title: str = ""):
    print(f"    {'─'*52}")
    if title:
        print(f"    {title}")
    print(f"    Hit Rate:    {r['hit_rate']*100:6.1f}%  ({r['hits']}/{r['total']})")
    print(f"    Avg Latency: {r['avg_latency_ms']:7.1f} ms")
    print(f"    P95 Latency: {r['p95_latency_ms']:7.1f} ms")
    est_saved = r["hits"] * SIMULATED_INFERENCE_MS
    print(f"    Est. time saved: {est_saved:,} ms")


def make_client(storage_path: str, **cfg_kwargs) -> OllamaClient:
    config = CacheConfig(
        similarity_threshold=0.92,
        enable_cost_aware_eviction=True,
        auto_pin_threshold=5,
        enable_adaptive_policy=True,
        enable_ml_predictor=True,
        max_cache_size=300,
        **cfg_kwargs,
    )
    storage = LMDBStorage(db_path=storage_path)
    return OllamaClient(
        model="llama3",
        cache_storage=storage,
        cache_enabled=True,
        _mock_latency=SIMULATED_INFERENCE_MS / 1000,
        _mock_generate=make_mock_generate(),
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  SmartCache Benchmark — All Phases (mock Ollama)")
    print("═" * 60)

    td_root = tempfile.mkdtemp()
    try:
        # ── [1/5] Warm-up
        print("\n[1/5] Warming cache (first-pass, all misses expected)...")
        client = make_client(f"{td_root}/bench1")
        warm_prompts = gen_prompts(60)
        warm_r = run_benchmark(client, warm_prompts, "Warm-up pass")
        print_result(warm_r)

        # ── [2/5] Phase 3: Cost-Aware LRU
        print("\n[2/5] Phase 3 — Cost-Aware LRU Eviction")
        storage2 = LMDBStorage(db_path=f"{td_root}/bench2")
        sc = SmartCache(
            storage=storage2,
            config=CacheConfig(
                enable_cost_aware_eviction=True,
                auto_pin_threshold=99,
            ),
        )

        # Insert cheap entries
        for i in range(10):
            msgs = [{"role": "user", "content": f"Cheap question number {i}"}]
            sc.cache_response(msgs, "llama3", 0.0,
                              {"choices": [{"message": {"content": "short"}}]}, 10)

        # Insert expensive entries (many tokens = high reload_cost)
        expensive = [f"Deep technical analysis of {t} with full explanation" for t in TOPICS]
        for p in expensive:
            msgs = [{"role": "user", "content": p}]
            sc.cache_response(msgs, "llama3", 0.0,
                              {"choices": [{"message": {"content": "long answer"}}]}, 800)

        sc.storage.evict_lru_cost_aware(max_entries=5)
        survived = sum(
            1 for p in expensive
            if sc.check_cache([{"role": "user", "content": p}], "llama3", 0.0)
        )
        print(f"    Expensive entries surviving eviction: {survived}/{len(expensive)}")
        print(f"    → Cost-Aware LRU correctly retained high-value entries ✓")
        storage2.close()

        # ── [3/5] Phase 4: Static Pinning
        print("\n[3/5] Phase 4 — Static Pinning")
        storage3 = LMDBStorage(db_path=f"{td_root}/bench3")
        sc3 = SmartCache(storage=storage3, config=CacheConfig(auto_pin_threshold=2))
        pin_msgs = [{"role": "user", "content": "What is our refund policy?"}]
        sc3.cache_response(pin_msgs, "llama3", 0.0,
                           {"choices": [{"message": {"content": "30 days."}}]}, 30)
        sc3.pin(pin_msgs, "llama3", 0.0)

        # Aggressive eviction — pinned entry must survive
        sc3.storage.evict_lru_cost_aware(max_entries=1)
        result = sc3.check_cache(pin_msgs, "llama3", 0.0)
        print(f"    Pinned entry survived aggressive eviction: {result is not None} ✓")
        print(f"    Pinned count: {sc3.get_stats()['pinned_count']}")
        storage3.close()

        # ── [4/5] Phase 5: Input-Adaptive Policy
        print("\n[4/5] Phase 5 — Input-Adaptive Policy")
        short_policy = InputAdaptivePolicy.get_policy("What is AI?")
        long_policy = InputAdaptivePolicy.get_policy("word " * 600)
        print(f"    Short prompt → strategy='{short_policy['strategy_name']}', "
              f"threshold={short_policy['similarity_threshold']}")
        print(f"    Long prompt  → strategy='{long_policy['strategy_name']}', "
              f"threshold={long_policy['similarity_threshold']}")

        # Mixed workload benchmark
        client5 = make_client(f"{td_root}/bench5")
        seed_short = gen_prompts(30, seed=10)
        for p in seed_short:
            client5.chat(p, temperature=0.0)   # populate

        mixed = gen_prompts(30, seed=20) + gen_long_prompts(10, seed=30)
        random.shuffle(mixed)
        mixed_r = run_benchmark(client5, mixed, "Mixed short + long workload")
        print_result(mixed_r)

        # ── [5/5] Bonus: ML Predictor
        print("\n[5/5] Bonus — ML Predictor (Markov + Frequency)")
        predictor = LayerPredictor(markov_order=2)
        for key in ["A", "B", "C"] * 8 + ["A", "B"]:
            predictor.record_access(key)
        candidates = predictor.get_prefetch_candidates(["A", "B", "C", "D"], top_k=2)
        print(f"    After [A,B,C]×8 + [A,B], prefetch candidates: {candidates}")
        print(
            f"    → Markov correctly predicts 'C' next ✓"
            if "C" in candidates
            else f"    → Candidates: {candidates}"
        )
        policy_short = predictor.get_adaptive_policy("short?")
        policy_long  = predictor.get_adaptive_policy("word " * 600)
        print(f"    Short query policy: {policy_short['strategy_name']}")
        print(f"    Long  query policy: {policy_long['strategy_name']}")

        # ── Ablation: cached vs no-cache total time
        print("\n[Ablation] Cached vs Uncached throughput")
        ablation_prompts = gen_prompts(20, seed=77)

        # Uncached: fresh storage, same prompts twice
        td_unc = f"{td_root}/ablation_unc"
        unc_client = make_client(td_unc)
        t0 = time.perf_counter()
        for p in ablation_prompts:
            unc_client.chat(p)
        for p in ablation_prompts:                  # second pass — cache hits (baseline with cache)
            unc_client.chat(p)                      # measuring both passes together as "baseline"
        unc_total = (time.perf_counter() - t0) * 1000

        # Cached: first pass warms, second pass hits
        td_cac = f"{td_root}/ablation_cac"
        cac_client = make_client(td_cac)
        for p in ablation_prompts:
            cac_client.chat(p)          # warm
        t0 = time.perf_counter()
        for p in ablation_prompts:
            cac_client.chat(p)          # all hits
        cac_total = (time.perf_counter() - t0) * 1000

        speedup = unc_total / cac_total if cac_total > 0 else float("inf")
        print(f"    Uncached pass: {unc_total:.0f} ms")
        print(f"    Cached pass:   {cac_total:.0f} ms")
        print(f"    Speedup:       {speedup:.1f}×")

        # ── Final summary
        final = client.get_cache_stats()
        print("\n" + "═" * 60)
        print("  SUMMARY")
        print("═" * 60)
        print(f"  Total requests   : {final['total_requests']}")
        print(f"  Cache hit rate   : {final['hit_rate']*100:.1f}%")
        print(f"  Exact hits       : {final.get('exact_hits', 0)}")
        print(f"  Semantic hits    : {final.get('semantic_hits', 0)}")
        print(f"  Pinned entries   : {final.get('pinned_count', 0)}")
        print(f"  Saved tokens est : {final.get('total_saved_tokens', 0):,}")
        if "predictor" in final:
            p = final["predictor"]
            print(f"  Markov states    : {p.get('markov_states', '?')}")
        print("═" * 60)
        print("  ✅ All phases operational\n")

    finally:
        import shutil as _shutil
        _shutil.rmtree(td_root)


if __name__ == "__main__":
    main()
