# 🏆 SmartCache Hackathon Submission

**SmartCache** is a Python module that adds semantic caching to local [Ollama](https://ollama.com) inference. It speeds up responses dramatically and cuts repeated inference work by understanding that "What is Python?" and "Explain Python" are the same question — no API keys or cloud dependencies required.

## The Problem

Local LLM inference can take 1–10 seconds per request.  If your app handles repetitive or paraphrased questions, you're paying that latency cost every single time.

## The Solution

```
Your App → OllamaClient → SmartCache → Ollama (only on cache miss)
```

SmartCache intercepts each prompt, encodes it as a semantic vector, and searches for a cached match.  Cache hits return in milliseconds.  Only genuine novel prompts reach Ollama.

## Quick Example

```python
from smartcache import OllamaClient

client = OllamaClient(model="llama3")

# First call — runs Ollama inference (~1-5s)
reply, meta = client.chat("What is Python?")
print(meta["cached"])          # False

# Second call — semantic match, instant cache hit
reply2, meta2 = client.chat("Explain Python programming")
print(meta2["cached"])         # True
print(meta2["time_saved_ms"])  # ~1500ms saved
```

## Four Advanced Phases

| Phase | Feature | Benefit |
|-------|---------|---------|
| Phase 3 | Cost-Aware LRU | Keeps expensive-to-regenerate entries longer |
| Phase 4 | Static Pinning | Critical FAQ entries are never evicted |
| Phase 5 | Input-Adaptive Policy | Threshold auto-adjusts to prompt length |
| Bonus  | ML Predictor | Markov + frequency model for proactive prefetch |

## Run It

```bash
ollama serve && ollama pull llama3
pip install -r requirements.txt
streamlit run demo/app.py        # live demo UI
python benchmark.py              # offline benchmark (no Ollama needed)
pytest tests/ -v                 # full test suite (offline)
```
