# ⚡ SmartCache v2

**Semantic caching for local Ollama inference — no API keys, no cloud, no cost.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SmartCache wraps your local [Ollama](https://ollama.com) models with a semantic caching layer.
Instead of repeating expensive inference for semantically identical prompts, SmartCache returns
cached answers in milliseconds.  v2 adds four advanced phases that make the cache smarter about
**what to keep, what to protect, and what to fetch next**.

---

## ✨ What's New in v2

| Phase  | Feature | Description |
|--------|---------|-------------|
| Phase 3 | **Cost-Aware LRU** | Evict based on `layer_size / reload_cost` — keep expensive responses longer |
| Phase 4 | **Static Pinning** | Mark critical entries as permanent; auto-pin hot entries past a hit threshold |
| Phase 5 | **Input-Adaptive Policy** | Dynamically adjust similarity threshold based on prompt length |
| Bonus  | **ML Predictor** | Markov chain + frequency model predicts next access for proactive prefetch |

---

## 🔧 Quick Start

### 1 — Start Ollama

```bash
ollama serve
ollama pull llama3      # or any other model
```

### 2 — Install SmartCache

```bash
pip install -r requirements.txt
```

### 3 — Use it

```python
from smartcache import OllamaClient

client = OllamaClient(model="llama3")

reply, meta = client.chat("What is Python?")
print(reply)
print("cache hit:", meta["cached"])       # False on first call

reply2, meta2 = client.chat("Explain Python")
print("cache hit:", meta2["cached"])      # True — semantic match!
print("time saved:", meta2["time_saved_ms"], "ms")
```

### Interactive model picker (CLI)

```python
from smartcache import OllamaClient
client = OllamaClient.from_interactive()   # shows a numbered menu of local models
```

### Streaming

```python
for chunk in client.chat_stream("Tell me a story about a clever robot"):
    print(chunk, end="", flush=True)
```

---

## 🖥️ Streamlit Demo

```bash
streamlit run demo/app.py
```

Opens at **http://localhost:8501**.  Features:
- Ollama health check banner + live model dropdown with refresh button
- Streaming chat with per-message HIT/MISS colour coding
- Cumulative time-saved chart
- Batch similarity test panel
- FAQ preload panel

---

## 📐 Phase Details

### Phase 3: Cost-Aware LRU

Standard LRU evicts the least recently used entry.  Cost-Aware LRU instead evicts the entry
with the lowest value-to-cost ratio:

```
eviction_score = (layer_size / reload_cost) × recency_weight
```

- `reload_cost` ∝ token count — expensive to regenerate → keep longer
- `layer_size` ∝ token count × hit frequency — high-use large entries score higher
- `recency_weight` = 1 / (1 + age_hours)

A short FAQ answer may be evicted before a 2000-token analysis even if both were accessed equally recently.

### Phase 4: Static Pinning

```python
# Manual pin — never evicted
client.pin_prompt("What is our refund policy?")

# Auto-pin via config — pins entries after N hits
client = OllamaClient(model="llama3", auto_pin_threshold=10)
```

### Phase 5: Input-Adaptive Policy

| Input length | Strategy | Threshold | Rationale |
|---|---|---|---|
| < 200 tokens | `ffn_broad` | 0.85 | Short queries have many paraphrases — be permissive |
| 200–500 tokens | `balanced` | 0.90 | Standard operation |
| > 500 tokens | `attention_strict` | 0.95 | Long, specific prompts require a close match |

Applied automatically when `enable_adaptive_policy=True` (default).

### Bonus: ML Predictor

Two complementary models combined:

**Markov Chain (order=2):** Learns sequential access patterns.
After observing `[A, B, C, A, B, C]`, predicts `C` will follow `A → B`.

**Frequency Decay:** Scores entries by time-decayed access frequency.
Hot entries bubble to the top even without sequential patterns.

---

## ⚙️ Full Configuration

```python
from smartcache import OllamaClient, SmartCache, CacheConfig
from smartcache.storage import LMDBStorage

client = OllamaClient(
    model="llama3",
    base_url="http://localhost:11434",    # default
    # Cache
    cache_enabled=True,
    similarity_threshold=0.92,
    cache_ttl=604800,                     # 7 days
    # Phase 4
    auto_pin_threshold=10,
    # Phase 5
    enable_adaptive_policy=True,
    # Bonus
    enable_ml_predictor=True,
)
```

Or compose directly with `SmartCache`:

```python
config = CacheConfig(
    similarity_threshold=0.92,
    enable_cost_aware_eviction=True,   # Phase 3
    auto_pin_threshold=10,             # Phase 4
    enable_adaptive_policy=True,       # Phase 5
    enable_ml_predictor=True,          # Bonus
    max_cache_size=10000,
)
cache = SmartCache(storage=LMDBStorage(), config=config)
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

All tests run offline — no Ollama server required.  Test classes:

- `TestEmbeddingEngine` — encode/similarity
- `TestCostAwareLRU` — eviction score ordering, pinned immunity, expensive-entry survival
- `TestStaticPinning` — pin/unpin, auto-pin on threshold
- `TestInputAdaptivePolicy` — strategy selection per input length
- `TestMarkovPredictor`, `TestFrequencyPredictor`, `TestLayerPredictor` — ML predictor
- `TestSmartCache` — integration: exact match, misses, stats, preload
- `TestOllamaClient` — chat, streaming, preload FAQ, pin, clear, health check
- `TestModelSelector` — filter, default, info parsing

## 📊 Benchmark

```bash
python benchmark.py
```

Runs a full ablation offline using a mock Ollama server, showing each phase's contribution
and a speedup measurement for cached vs uncached throughput.

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌───────────────────────────┐    ┌──────────────────┐
│   Your App      │───▶│       OllamaClient        │───▶│  Ollama Server   │
└─────────────────┘    └───────────────────────────┘    │  (local, free)   │
                                    │                    └──────────────────┘
                                    ▼
                       ┌────────────────────────────┐
                       │         SmartCache         │
                       │                            │
                       │  ┌──────────────────────┐  │
                       │  │  InputAdaptivePolicy │  │  ← Phase 5
                       │  └──────────────────────┘  │
                       │  ┌──────────────────────┐  │
                       │  │   EmbeddingEngine    │  │
                       │  └──────────────────────┘  │
                       │  ┌──────────────────────┐  │
                       │  │  LayerPredictor      │  │  ← Bonus ML
                       │  │  (Markov+Frequency)  │  │
                       │  └──────────────────────┘  │
                       └────────────┬───────────────┘
                                    │
                       ┌────────────▼───────────────┐
                       │      LMDBStorage           │
                       │  - Cost-Aware LRU eviction │  ← Phase 3
                       │  - Static pinning          │  ← Phase 4
                       └────────────────────────────┘
```

---

## 🚫 What was removed

SmartCache v2 no longer has any OpenAI dependency.  The `CachedOpenAI` wrapper and
`openai_wrapper` module have been removed.  Everything now runs locally via Ollama.
