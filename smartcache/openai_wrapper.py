"""
openai_wrapper.py — REMOVED

SmartCache no longer depends on or wraps OpenAI.
Use OllamaClient for local inference with semantic caching.

    from smartcache import OllamaClient
    client = OllamaClient(model="llama3")
    reply, meta = client.chat("What is Python?")
"""

raise ImportError(
    "openai_wrapper has been removed.  "
    "Use 'from smartcache import OllamaClient' instead."
)
