"""
SmartCache Demo — Streamlit UI for local Ollama inference with semantic caching.

No API keys required.  Just run:
    ollama serve
    streamlit run demo/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartcache import OllamaClient
from smartcache.storage import LMDBStorage

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartCache · Ollama",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.hit-badge  { color: #00c853; font-weight: bold; }
.miss-badge { color: #e53935; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ SmartCache")
st.caption("Semantic caching for local Ollama inference — no API keys, no cost, just speed.")

# ─────────────────────────────────────────────────────────────
# Sidebar — collect all inputs first
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    ollama_url = st.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="Where your Ollama server is running",
    )

    # Health check
    healthy = OllamaClient.health_check(ollama_url)
    if healthy:
        st.success("✅ Ollama is running")
    else:
        st.error("❌ Ollama unreachable — start with `ollama serve`")

    # Model picker
    st.subheader("Model")
    col_model, col_refresh = st.columns([3, 1])
    with col_refresh:
        do_refresh = st.button("🔄", help="Refresh model list")

    if "model_list" not in st.session_state or do_refresh:
        st.session_state.model_list = OllamaClient.list_models(ollama_url) or ["llama3"]

    selected_model = col_model.selectbox(
        "Model",
        st.session_state.model_list,
        label_visibility="collapsed",
    )

    # Cache settings
    st.subheader("Cache settings")
    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.70,
        max_value=0.99,
        value=0.92,
        step=0.01,
        help="Higher = stricter matching",
    )
    enable_adaptive = st.checkbox("Adaptive policy (Phase 5)", value=True)
    enable_ml = st.checkbox("ML predictor (Bonus)", value=True)

    st.divider()
    # Capture clear intent — client not available yet, handled below
    do_clear = st.button("🗑️ Clear cache", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Client — initialised AFTER sidebar inputs are collected
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_client(model: str, url: str, threshold: float, adaptive: bool, ml: bool):
    storage = LMDBStorage(db_path="./smartcache_demo_data")
    return OllamaClient(
        model=model,
        base_url=url,
        cache_storage=storage,
        similarity_threshold=threshold,
        enable_adaptive_policy=adaptive,
        enable_ml_predictor=ml,
    )


client = get_client(
    selected_model, ollama_url, similarity_threshold, enable_adaptive, enable_ml
)

# Handle clear now that client exists
if do_clear:
    client.clear_cache()
    for key in ["history", "time_saved_history"]:
        st.session_state.pop(key, None)
    st.sidebar.success("Cache cleared")

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "time_saved_history" not in st.session_state:
    st.session_state.time_saved_history = []

# ─────────────────────────────────────────────────────────────
# Stats dashboard
# ─────────────────────────────────────────────────────────────

st.subheader("📊 Live Statistics")

stats = client.get_cache_stats()
total        = stats.get("total_requests", 0)
hits         = stats.get("hits", 0)
misses       = stats.get("misses", 0)
hit_rate     = stats.get("hit_rate", 0.0) * 100
total_saved_ms = sum(st.session_state.time_saved_history)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests",      total)
c2.metric("Hits",          hits)
c3.metric("Misses",        misses)
c4.metric("Hit Rate",      f"{hit_rate:.1f}%")
c5.metric("Est. Time Saved", f"{total_saved_ms / 1000:.1f}s")

if total > 0:
    st.progress(hits / total, text=f"Cache efficiency: {hit_rate:.1f}%")

# ─────────────────────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────────────────────

st.subheader("💬 Chat")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["prompt"])
    with st.chat_message("assistant"):
        badge = (
            '<span class="hit-badge">⚡ Cache HIT</span>'
            if turn["cached"]
            else '<span class="miss-badge">🔄 Cache MISS</span>'
        )
        st.markdown(badge, unsafe_allow_html=True)
        st.write(turn["reply"])
        detail = f"⏱ {turn['latency_ms']:.0f} ms"
        if turn["cached"] and turn.get("time_saved_ms", 0) > 0:
            detail += f" · saved ~{turn['time_saved_ms']:.0f} ms"
        if turn.get("similarity") is not None:
            detail += f" · similarity {turn['similarity']:.3f}"
        st.caption(detail)

if prompt := st.chat_input("Ask anything…", disabled=not healthy):
    with st.spinner("Querying…"):
        reply, meta = client.chat(prompt, temperature=0.0)

    turn = {
        "prompt": prompt,
        "reply": reply,
        "cached": meta["cached"],
        "latency_ms": meta["latency_ms"],
        "time_saved_ms": meta.get("time_saved_ms", 0.0),
        "similarity": meta.get("similarity"),
    }
    st.session_state.history.append(turn)
    if meta["cached"]:
        st.session_state.time_saved_history.append(meta.get("time_saved_ms", 0.0))
    st.rerun()

# ─────────────────────────────────────────────────────────────
# Batch test panel
# ─────────────────────────────────────────────────────────────

with st.expander("🧪 Batch Similarity Test"):
    st.caption("Send multiple prompts at once to visualise hit/miss patterns.")
    batch_input = st.text_area(
        "Prompts (one per line)",
        value=(
            "What is Python?\n"
            "Explain Python programming\n"
            "How do I learn Python?\n"
            "What is machine learning?\n"
            "Explain ML to me\n"
            "Tell me about deep learning"
        ),
        height=130,
    )

    if st.button("▶ Run Batch", type="primary", disabled=not healthy):
        prompts = [p.strip() for p in batch_input.strip().splitlines() if p.strip()]
        results = []
        prog = st.progress(0)

        for i, p in enumerate(prompts):
            reply, meta = client.chat(p, temperature=0.0)
            results.append({
                "Prompt":          p[:55] + ("…" if len(p) > 55 else ""),
                "Status":          "✅ HIT" if meta["cached"] else "❌ MISS",
                "Latency (ms)":    round(meta["latency_ms"], 1),
                "Time Saved (ms)": round(meta.get("time_saved_ms", 0.0)),
                "Hit Type":        meta.get("hit_type", "miss"),
            })
            if meta["cached"]:
                st.session_state.time_saved_history.append(meta.get("time_saved_ms", 0.0))
            prog.progress((i + 1) / len(prompts))

        prog.empty()
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df,
            x="Prompt",
            y="Latency (ms)",
            color="Status",
            color_discrete_map={"✅ HIT": "#00c853", "❌ MISS": "#e53935"},
            title="Latency per Prompt",
        )
        fig.update_layout(height=320, margin=dict(t=40, b=60))
        st.plotly_chart(fig, use_container_width=True)
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Time-saved chart
# ─────────────────────────────────────────────────────────────

if len(st.session_state.time_saved_history) >= 2:
    with st.expander("⏱ Time Saved Over Cache Hits", expanded=True):
        cumulative = []
        running = 0.0
        for ms in st.session_state.time_saved_history:
            running += ms
            cumulative.append(running)

        fig2 = go.Figure(
            go.Scatter(
                y=[v / 1000 for v in cumulative],
                mode="lines+markers",
                line=dict(color="#00c853", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,200,83,0.1)",
                name="Cumulative seconds saved",
            )
        )
        fig2.update_layout(
            title="Cumulative Time Saved (seconds)",
            xaxis_title="Cache Hit #",
            yaxis_title="Seconds",
            height=280,
            margin=dict(t=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FAQ preload panel
# ─────────────────────────────────────────────────────────────

with st.expander("📚 Preload FAQ"):
    st.caption("Format:  question|answer  (one per line)")
    faq_text = st.text_area(
        "FAQ pairs",
        value=(
            "What is SmartCache?|SmartCache is a semantic caching layer for local Ollama inference.\n"
            "How does semantic caching work?|It uses sentence embeddings to find similar prompts.\n"
            "What speed improvement can I expect?|Cache hits return in milliseconds vs seconds.\n"
            "Does this need an API key?|No. SmartCache works entirely with local Ollama models."
        ),
        height=120,
        label_visibility="collapsed",
    )
    if st.button("📥 Load FAQ"):
        pairs = []
        for line in faq_text.strip().splitlines():
            if "|" in line:
                q, a = line.split("|", 1)
                pairs.append((q.strip(), a.strip(), 0.0, 30))
        client.preload_faq(pairs)
        st.success(f"Loaded {len(pairs)} FAQ entries into cache")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**SmartCache** · Phase 3: Cost-Aware LRU · Phase 4: Static Pinning · "
    "Phase 5: Adaptive Policy · Bonus: ML Predictor"
)
