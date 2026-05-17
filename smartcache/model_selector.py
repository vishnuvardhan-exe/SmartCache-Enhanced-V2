"""
ModelSelector — interactive and programmatic Ollama model selection.

Provides a thin wrapper around OllamaClient.list_models() with
extra filtering, display, and selection helpers.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class ModelSelector:
    """
    Choose an Ollama model interactively or by filter.

    Example::

        selector = ModelSelector()
        model = selector.pick()              # interactive CLI
        model = selector.pick(filter="llama")  # first match for filter
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self._models: Optional[List[str]] = None

    def refresh(self) -> List[str]:
        """Fetch fresh model list from Ollama."""
        from .ollama_client import OllamaClient
        self._models = OllamaClient.list_models(self.base_url)
        return self._models

    def available(self) -> List[str]:
        """Return cached model list, fetching if needed."""
        if self._models is None:
            self.refresh()
        return self._models or []

    def info(self) -> List[Dict[str, Any]]:
        """
        Return a list of dicts with basic size info parsed from model tags.
        Tags use the convention  <name>:<size>  e.g. ``llama3:8b``.
        """
        result = []
        for name in self.available():
            parts = name.split(":")
            tag = parts[1] if len(parts) > 1 else "latest"
            result.append({"name": name, "base": parts[0], "tag": tag})
        return result

    def pick(
        self,
        *,
        filter: Optional[str] = None,
        default: Optional[str] = None,
        interactive: bool = True,
    ) -> str:
        """
        Select a model.

        Args:
            filter: If given, return the first model whose name contains this string.
            default: Fallback if filter matches nothing and interactive is False.
            interactive: If True and no filter match, prompt the user on the CLI.

        Returns:
            Model name string.
        """
        models = self.available()
        if not models:
            raise RuntimeError(
                "No models available.  Is Ollama running?  Start with:  ollama serve"
            )

        if filter:
            matches = [m for m in models if filter.lower() in m.lower()]
            if matches:
                return matches[0]
            logger.warning("No model matching '%s'; falling through", filter)

        if not interactive:
            if default and default in models:
                return default
            return models[0]

        print("\nAvailable Ollama models:")
        for i, name in enumerate(models, 1):
            print(f"  {i:>3}. {name}")
        print()

        while True:
            raw = input(f"Select [1-{len(models)}] or type model name: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(models):
                    return models[idx]
            except ValueError:
                if raw in models:
                    return raw
            print("  Invalid — try again.")

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._models) if self._models is not None else "?"
        return f"ModelSelector(base_url={self.base_url!r}, models={n})"
