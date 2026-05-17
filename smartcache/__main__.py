"""
SmartCache: Semantic Caching for local Ollama inference

Usage:
    python -m smartcache demo
        # Runs the Streamlit demo

    Or use programmatically:
        from smartcache import OllamaClient
        client = OllamaClient(model="llama3")
        reply, meta = client.chat("What is Python?")
        print(reply)
"""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        import subprocess
        import os
        demo_path = os.path.join(os.path.dirname(__file__), "..", "demo", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", demo_path])
    else:
        print(__doc__)
