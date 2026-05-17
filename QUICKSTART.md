# ⚡ SmartCache Quick Start

Get the demo running in under 5 minutes!

## Prerequisites

1. **Install Ollama** — https://ollama.com
2. **Start Ollama** and pull a model:
   ```bash
   ollama serve
   ollama pull llama3   # or: mistral, phi3, gemma3, etc.
   ```

---

## Option 1: One-Command (Recommended)

```bash
./run_demo.sh
```

Then open **http://localhost:8501**

---

## Option 2: Manual Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit demo
streamlit run demo/app.py
```

---

## Option 3: Docker

```bash
docker-compose up -d
# Open http://localhost:8501
```

---

## Using the Demo

1. The sidebar shows a **health check** — green means Ollama is running.
2. Pick your model from the **dropdown** (click 🔄 to refresh).
3. Type a message in the **chat box** and hit Enter.
4. Send a semantically similar message — you'll see a **⚡ Cache HIT** in milliseconds!
5. Use the **Batch Test** panel to send several prompts at once and see the latency chart.

---

## Try These Prompts

```
What is Python?
Explain Python programming    ← should hit the first one
How does Python work?         ← should also hit

What is machine learning?
Explain ML                    ← semantic hit
```

You'll see:
- First prompt: ❌ **Cache MISS** — inference runs (~1–5s depending on model)
- Similar prompts: ✅ **Cache HIT** — returned instantly, inference skipped

---

## Basic Code Example

```python
from smartcache import OllamaClient

client = OllamaClient(model="llama3")

reply, meta = client.chat("What is Python?")
print(reply)
print("cached:", meta["cached"])          # False

reply2, meta2 = client.chat("Explain Python")
print("cached:", meta2["cached"])         # True — semantic match
print("saved:", meta2["time_saved_ms"], "ms")
```

---

## Interactive Model Picker

```python
from smartcache import OllamaClient
client = OllamaClient.from_interactive()   # choose from numbered list
reply, meta = client.chat("Hello!")
```

---

## Run Tests & Benchmark (offline — no Ollama needed)

```bash
pytest tests/ -v
python benchmark.py
python example.py
```

---

## Need Help?

- Full docs: [README.md](README.md)
- Run tests: `make test`
- Example script: `python example.py`

---

**👋 That's it! Enjoy caching!**
