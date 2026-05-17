"""
Ollama client with SmartCache semantic caching integration.

Wraps local Ollama inference with full SmartCache support:
- Semantic cache lookup before every generate call
- Cache-on-miss with token count tracking
- Time-saved reporting (no API keys or costs needed)
- Interactive model picker via from_interactive()
"""

from typing import Optional, Dict, Any, List, Iterator
import json
import logging
import time
import urllib.request
import urllib.error

from .core import SmartCache, CacheConfig
from .storage import LMDBStorage

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


def _http_get(url: str, timeout: float = 5.0) -> Any:
    """Simple HTTP GET returning parsed JSON, or raises on error."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post(url: str, payload: Dict, timeout: float = 120.0) -> Any:
    """Simple HTTP POST with JSON body, returns parsed JSON."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post_stream(url: str, payload: Dict, timeout: float = 120.0) -> Iterator[str]:
    """HTTP POST with NDJSON streaming; yields each decoded text chunk."""
    data = json.dumps({**payload, "stream": True}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "") or obj.get("response", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break
            except json.JSONDecodeError:
                continue


class OllamaClient:
    """
    Ollama chat client with SmartCache semantic caching.

    Example (basic)::

        client = OllamaClient(model="llama3")
        reply, meta = client.chat("What is Python?")
        print(reply)
        print("cache hit:", meta["cached"])

    Example (streaming)::

        for chunk in client.chat_stream("Tell me a story"):
            print(chunk, end="", flush=True)

    Example (interactive CLI model picker)::

        client = OllamaClient.from_interactive()
    """

    def __init__(
        self,
        model: str = "llama3",
        *,
        base_url: str = OLLAMA_BASE_URL,
        cache_enabled: bool = True,
        cache_storage=None,
        similarity_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_ttl: int = 604800,
        enable_adaptive_policy: bool = True,
        enable_ml_predictor: bool = True,
        auto_pin_threshold: int = 10,
        # Simulated inference latency for benchmarking (seconds); 0 = real HTTP
        _mock_latency: float = 0.0,
        # Optional callable used to mock HTTP responses in tests
        _mock_generate=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_enabled = cache_enabled
        self._mock_latency = _mock_latency
        self._mock_generate = _mock_generate

        if cache_enabled:
            config = CacheConfig(
                similarity_threshold=similarity_threshold,
                embedding_model=embedding_model,
                cache_ttl=cache_ttl,
                enable_adaptive_policy=enable_adaptive_policy,
                enable_ml_predictor=enable_ml_predictor,
                auto_pin_threshold=auto_pin_threshold,
            )
            storage = cache_storage or LMDBStorage(db_path="./smartcache_ollama_data")
            self.cache = SmartCache(storage=storage, config=config)
        else:
            self.cache = None

        logger.info(
            "OllamaClient initialised: model=%s cache=%s threshold=%.2f",
            model,
            "enabled" if cache_enabled else "disabled",
            similarity_threshold,
        )

    # ─────────────────────────────────────────────────────────
    # Class-level helpers
    # ─────────────────────────────────────────────────────────

    @classmethod
    def list_models(cls, base_url: str = OLLAMA_BASE_URL) -> List[str]:
        """Return sorted list of model names available on the local Ollama server."""
        try:
            data = _http_get(f"{base_url}/api/tags", timeout=5.0)
            return sorted(m["name"] for m in data.get("models", []))
        except Exception as exc:
            logger.warning("Could not reach Ollama at %s: %s", base_url, exc)
            return []

    @classmethod
    def health_check(cls, base_url: str = OLLAMA_BASE_URL) -> bool:
        """Return True if Ollama is reachable."""
        try:
            _http_get(f"{base_url}/api/tags", timeout=3.0)
            return True
        except Exception:
            return False

    @classmethod
    def from_interactive(cls, base_url: str = OLLAMA_BASE_URL, **kwargs) -> "OllamaClient":
        """
        Interactive CLI model picker.  Lists available Ollama models and
        prompts the user to choose one, then returns a configured OllamaClient.

        Usage::

            client = OllamaClient.from_interactive()
        """
        if not cls.health_check(base_url):
            raise RuntimeError(
                f"Ollama is not running at {base_url}.\n"
                "Start it with:  ollama serve"
            )

        models = cls.list_models(base_url)
        if not models:
            raise RuntimeError("No models found.  Pull one first, e.g.:  ollama pull llama3")

        print("\nAvailable Ollama models:")
        for i, name in enumerate(models, 1):
            print(f"  {i:>2}. {name}")

        while True:
            raw = input(f"\nSelect model [1-{len(models)}]: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(models):
                    chosen = models[idx]
                    break
            except ValueError:
                # Accept typing the name directly
                if raw in models:
                    chosen = raw
                    break
            print("  Invalid choice, try again.")

        print(f"  ✓ Using model: {chosen}\n")
        return cls(model=chosen, base_url=base_url, **kwargs)

    # ─────────────────────────────────────────────────────────
    # Core generate (internal)
    # ─────────────────────────────────────────────────────────

    def _generate(self, messages: List[Dict]) -> Dict:
        """Call Ollama /api/chat or mock; returns raw response dict."""
        if self._mock_generate is not None:
            if self._mock_latency:
                time.sleep(self._mock_latency)
            return self._mock_generate(messages, self.model)

        payload = {"model": self.model, "messages": messages, "stream": False}
        return _http_post(f"{self.base_url}/api/chat", payload)

    def _generate_stream(self, messages: List[Dict]) -> Iterator[str]:
        """Yield streaming text chunks from Ollama."""
        payload = {"model": self.model, "messages": messages}
        yield from _http_post_stream(f"{self.base_url}/api/chat", payload)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> tuple:  # (reply: str, metadata: dict)
        """
        Send a single user message and return (reply, metadata).

        metadata keys:
            cached (bool), latency_ms (float), model (str),
            hit_type ("exact"|"semantic"|"miss"),
            time_saved_ms (float)  — estimated ms saved vs a real inference call.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()

        # Cache lookup
        if self.cache_enabled and self.cache:
            cached = self.cache.check_cache(messages, self.model, temperature)
            if cached:
                cached_resp, cache_meta = cached
                latency_ms = (time.perf_counter() - t0) * 1000
                reply = (
                    cached_resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or cached_resp.get("response", "")
                    or str(cached_resp)
                )
                return reply, {
                    "cached": True,
                    "hit_type": cache_meta.get("hit_type", "semantic"),
                    "latency_ms": latency_ms,
                    # avg_inference_ms is stored inside the response payload at cache time
                    "time_saved_ms": cached_resp.get("avg_inference_ms", 1500.0),
                    "model": self.model,
                    "similarity": cache_meta.get("similarity", 1.0),
                }

        # Real inference
        raw = self._generate(messages)
        latency_ms = (time.perf_counter() - t0) * 1000
        reply = (
            raw.get("message", {}).get("content", "")
            or raw.get("response", "")
        )
        token_count = (
            raw.get("eval_count", 0) + raw.get("prompt_eval_count", 0)
        ) or max(1, len(reply.split()) * 4 // 3)

        # Store in cache with inferred inference time for future time_saved reporting
        if self.cache_enabled and self.cache:
            response_payload = {
                "choices": [{"message": {"content": reply}}],
                "response": reply,
                "avg_inference_ms": latency_ms,
            }
            self.cache.cache_response(
                messages, self.model, temperature, response_payload, token_count
            )

        return reply, {
            "cached": False,
            "hit_type": "miss",
            "latency_ms": latency_ms,
            "time_saved_ms": 0.0,
            "model": self.model,
            "token_count": token_count,
        }

    def chat_stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """
        Yield streaming text chunks.  Cache hits are returned as a single chunk.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.cache_enabled and self.cache:
            cached = self.cache.check_cache(messages, self.model, temperature)
            if cached:
                cached_resp, _ = cached
                reply = (
                    cached_resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or cached_resp.get("response", "")
                    or str(cached_resp)
                )
                yield reply
                return

        # Stream from Ollama and accumulate for caching
        chunks = []
        for chunk in self._generate_stream(messages):
            chunks.append(chunk)
            yield chunk

        full_reply = "".join(chunks)
        if self.cache_enabled and self.cache and full_reply:
            token_count = max(1, len(full_reply.split()) * 4 // 3)
            self.cache.cache_response(
                messages,
                self.model,
                temperature,
                {"choices": [{"message": {"content": full_reply}}], "response": full_reply},
                token_count,
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return full SmartCache statistics dict."""
        if self.cache:
            return self.cache.get_stats()
        return {"error": "Cache not enabled"}

    def pin_prompt(self, prompt: str, *, temperature: float = 0.0):
        """Phase 4: Pin a prompt so its cached response is never evicted."""
        if self.cache:
            messages = [{"role": "user", "content": prompt}]
            self.cache.pin(messages, self.model, temperature)
            logger.info("Pinned: %s…", prompt[:60])

    def preload_faq(self, faq_pairs: List[tuple]):
        """
        Preload cache with FAQ pairs.

        Args:
            faq_pairs: list of (question, answer, temperature, token_count)
                       or (question, answer) using defaults.
        """
        if not self.cache:
            return
        for row in faq_pairs:
            if len(row) == 2:
                q, a = row
                temp, tokens = 0.0, 50
            else:
                q, a, temp, tokens = row[0], row[1], row[2] if len(row) > 2 else 0.0, row[3] if len(row) > 3 else 50
            messages = [{"role": "user", "content": q}]
            self.cache.cache_response(
                messages,
                self.model,
                temp,
                {"choices": [{"message": {"content": a}}], "response": a},
                tokens,
            )
        logger.info("Preloaded %d FAQ entries", len(faq_pairs))

    def clear_cache(self):
        """Clear all cached entries."""
        if self.cache:
            self.cache.clear()

    def get_prefetch_candidates(self) -> List[str]:
        """Bonus ML: return cache keys predicted to be accessed soon."""
        if self.cache:
            return self.cache.get_prefetch_candidates()
        return []
