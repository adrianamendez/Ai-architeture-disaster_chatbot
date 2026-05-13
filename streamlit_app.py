"""Streamlit dashboard for the Natural Disaster Intelligence Chatbot project.

Run:
    streamlit run streamlit_app.py

This dashboard does NOT call Ollama, NASA, GDACS, or load any model. It renders
the kind of output the notebook (`disaster_chatbot.ipynb`) produces, using the
fixtures in `streamlit_data.py`. It exists to demo the project end-to-end without
needing the local LLM stack to be running.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import streamlit_data as DATA

# Image folders (relative to this script so it works on Streamlit Cloud too).
COLOMBIA_IMAGES_DIR = Path(__file__).parent / "test_images_colombia"
SAMPLE_IMAGES_DIR = Path(__file__).parent / "sample_images"

# -----------------------------------------------------------------------------
# Page config + global CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Disaster Intelligence Chatbot",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
        padding: 18px; border-radius: 8px; color: white;
        text-align: center; min-height: 110px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; opacity: 0.85; font-weight: 500; }
    .metric-card h1 { margin: 6px 0 0; font-size: 32px; font-weight: 700; }
    .metric-card p  { margin: 4px 0 0; font-size: 11px; opacity: 0.75; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
             font-size: 11px; font-weight: 600; }
    .badge-pass { background: #e8f5e9; color: #1b5e20; }
    .badge-fail { background: #ffebee; color: #b71c1c; }
    .small-caption { color: #888; font-size: 12px; }
    .section-title { border-left: 4px solid #1976d2; padding-left: 10px; margin-top: 8px; }
    .hero {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 50%, #c62828 100%);
        padding: 36px 28px; border-radius: 12px; color: white; margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .hero h1 { color: white; margin: 0; font-size: 32px; }
    .hero p  { color: rgba(255,255,255,0.92); margin: 8px 0 0; font-size: 15px; }
    .hero .tag { display: inline-block; background: rgba(255,255,255,0.18);
                 padding: 3px 10px; border-radius: 12px; font-size: 12px;
                 margin-right: 6px; }
    .key-finding {
        background: #fff3e0; border-left: 4px solid #ef6c00;
        padding: 12px 16px; border-radius: 4px; margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div class='metric-card'>
              <h3>{label}</h3><h1>{value}</h1><p>{sub}</p>
           </div>""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌋 Disaster Intelligence")
    st.caption("Final Project — RAG + MCP + Agents")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Data Exploration",
            "🤖 Agent Demo",
            "🖼️ Image Classification",
            "🌎 Real-World Test (Colombia)",
            "📈 Evaluation",
            "🛡️ Security",
            "📝 Conclusions",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Stack**")
    for k, v in DATA.PROJECT["stack"].items():
        st.markdown(f"- **{k}** — {v}")
    st.divider()
    st.caption(
        "This dashboard renders pre-computed results from the notebook. "
        "It does not call Ollama or any external API."
    )


# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------

def page_overview():
    # ---- Hero ---------------------------------------------------------------
    st.markdown(
        f"""
        <div class='hero'>
          <h1>🌋 {DATA.PROJECT['title']}</h1>
          <p>{DATA.PROJECT['subtitle']}</p>
          <p style='margin-top:14px;'>
            <span class='tag'>RAG + reranking</span>
            <span class='tag'>ReAct agent</span>
            <span class='tag'>MCP server</span>
            <span class='tag'>Multimodal (LLaVA)</span>
            <span class='tag'>Live feeds</span>
            <span class='tag'>OWASP-aware</span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Visual gallery: what the chatbot reasons about ---------------------
    st.markdown("#### What this chatbot reasons about")
    classes_order = ["fire", "flood", "landslide", "earthquake", "smoke", "normal"]
    cols = st.columns(6)
    for col, cls in zip(cols, classes_order):
        img_path = SAMPLE_IMAGES_DIR / f"{cls}.jpg"
        with col:
            if img_path.exists():
                st.image(str(img_path), caption=cls.upper(), use_container_width=True)
    st.caption(
        "Six visual classes the multimodal stack distinguishes (LLaVA via Ollama), "
        "trained-on / evaluated against ~9k labeled images from Kaggle. "
        "On top of imagery the system also reasons over 80K+ historical disaster records "
        "and live event feeds (NASA EONET + GDACS)."
    )

    st.divider()

    # ---- Headline numbers ---------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Datasets", "3 + image set", "Kaggle / EM-DAT")
    with c2: metric_card("MCP tools", "8", "FastMCP over stdio")
    with c3: metric_card("Live feeds", "2", "NASA EONET + GDACS")
    with c4: metric_card("Tests", "75 / 75", "pytest passing")

    # ---- Project lineage (matches notebook §0 + §0.5) -----------------------
    st.markdown("### Project lineage")
    st.caption("This final project consolidates three previous course works into a single end-to-end system.")
    for proj, contribution in DATA.PROJECT["lineage"]:
        with st.container(border=True):
            st.markdown(f"**`{proj}`** — {contribution}")

    # ---- Architecture (matches notebook intro ASCII diagram) ----------------
    st.markdown("### Architecture")
    c_diag, c_layers = st.columns([2, 1])
    with c_diag:
        st.code(
            """
                ┌─────────────────────────────────────────┐
                │   User (Streamlit / Notebook)           │
                └──────────────────┬──────────────────────┘
                                   ▼
                ┌─────────────────────────────────────────┐
                │   LangChain ReAct Agent  (llama3.2)     │
                │   ─ decides which tool to use per query │
                └─────┬──────┬───────┬───────┬────────────┘
                      ▼      ▼       ▼       ▼
              ┌────────┐┌──────┐┌──────┐ ┌────────┐ ┌──────┐
              │ RAG +  ││ CSV  ││ NASA │ │ GDACS  │ │LLaVA │
              │ rerank ││ tool ││ EONET│ │humanit.│ │vision│
              └───┬────┘└──────┘└──────┘ └────────┘ └──────┘
                  │
                  └─> ChromaDB + MiniLM + cross-encoder
            """,
            language=None,
        )
    with c_layers:
        st.markdown("**Stack at a glance**")
        for k, v in DATA.PROJECT["stack"].items():
            st.markdown(f"- **{k}** — `{v}`")

    # ---- Why this project (academic framing) --------------------------------
    st.markdown("### Why this project")
    st.markdown(
        "Disaster response decisions sit on a knife edge: hours of delay translate into "
        "lives. Yet the relevant information is fragmented across **historical statistics** "
        "(EM-DAT, multi-decade CSVs), **live geospatial feeds** (NASA EONET, GDACS), and "
        "**unstructured visual evidence** (eyewitness photos). This project demonstrates "
        "that a single tool-using LLM agent — running entirely on local hardware via Ollama — "
        "can integrate all three modalities behind a uniform Q&A interface, with measurable "
        "and reproducible quality."
    )

    # ---- Data + endpoint inventory ------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Datasets used")
        st.dataframe(
            pd.DataFrame(DATA.PROJECT["datasets"], columns=["Dataset", "Source"]),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown("### Live endpoints consumed")
        st.dataframe(
            pd.DataFrame(DATA.PROJECT["live_endpoints"], columns=["Service", "Endpoint"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown(
        "<div class='key-finding'><b>Key finding (preview):</b> the ReAct agent picks the "
        "correct tool in 92% of benchmark queries (vs. ~25% for a uniform-random baseline) and "
        "the RAG pipeline with cross-encoder reranking achieves 80% context precision on "
        "RAG-routed questions. Full numbers in the <i>Evaluation</i> page.</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page: Data Exploration
# -----------------------------------------------------------------------------

def page_data():
    st.title("📊 Data Exploration")
    st.caption("Snapshot of the historical disaster corpus the chatbot reasons over (~80K records).")

    # ---- Visual taste of the corpus ----
    with st.expander("🖼️ Visual sample of the image corpus (one per class)", expanded=False):
        classes_order = ["fire", "flood", "landslide", "earthquake", "smoke", "normal"]
        cols = st.columns(6)
        for col, cls in zip(cols, classes_order):
            img_path = SAMPLE_IMAGES_DIR / f"{cls}.jpg"
            with col:
                if img_path.exists():
                    st.image(str(img_path), caption=cls.upper(), use_container_width=True)
        st.caption("These are the 6 classes the LLaVA classifier in Section 5 has to distinguish.")

    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame(
            sorted(DATA.DISASTER_TYPE_COUNTS.items(), key=lambda x: -x[1]),
            columns=["Disaster type", "Events"],
        )
        fig = px.bar(df, x="Events", y="Disaster type", orientation="h",
                     color="Events", color_continuous_scale="Blues",
                     title="Top disaster types by frequency")
        fig.update_layout(height=380, coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df = pd.DataFrame(
            sorted(DATA.CASUALTIES_BY_TYPE.items(), key=lambda x: -x[1]),
            columns=["Disaster type", "Casualties"],
        )
        fig = px.bar(df, x="Casualties", y="Disaster type", orientation="h",
                     color="Casualties", color_continuous_scale="Reds",
                     title="Total casualties by disaster type")
        fig.update_layout(height=380, coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Casualties by country (choropleth)")
    geo_df = pd.DataFrame(
        list(DATA.COUNTRY_CASUALTIES.items()), columns=["country", "casualties"]
    )
    fig = px.choropleth(
        geo_df, locations="country", locationmode="country names",
        color="casualties", color_continuous_scale="Reds",
        hover_name="country",
        title="Total disaster casualties by country (top 20)",
    )
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Reading the map:** China and India dominate by absolute casualties, partly because "
        "of population density. Haiti stands out for its size — a single event (the 2010 "
        "earthquake) drives most of that bar. **Caveat:** EM-DAT historically over-represents "
        "countries with strong reporting infrastructure, so 'absence of red' here does not "
        "mean 'no disasters'."
    )

    st.markdown("### Events over time (top 5 disaster types)")
    timeline_df = pd.DataFrame(DATA.TIMELINE)
    fig = px.line(timeline_df, x="year", y="count", color="type", markers=True,
                  title="Disaster events per year, by type")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Page: Agent Demo
# -----------------------------------------------------------------------------

def page_agent():
    st.title("🤖 Agent Demo — Interactive Chat")
    st.caption(
        "Five representative queries (one per tool category). The agent routes "
        "each question to the appropriate read-only tool. Answers below are "
        "cached responses from a notebook run — no live LLM call here."
    )

    # Tool routing visualization (Sankey)
    st.markdown("### Agent routing")
    categories = sorted({t["category"] for t in DATA.AGENT_TURNS})
    tools_set = sorted({tool for t in DATA.AGENT_TURNS for tool in t["tools_used"]})
    labels = categories + tools_set
    cat_idx = {c: i for i, c in enumerate(categories)}
    tool_idx = {t: len(categories) + i for i, t in enumerate(tools_set)}

    flow = Counter()
    for turn in DATA.AGENT_TURNS:
        for tool in turn["tools_used"]:
            flow[(turn["category"], tool)] += 1

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=18,
                  color=["#1976d2"] * len(categories) + ["#ff9800"] * len(tools_set)),
        link=dict(
            source=[cat_idx[c] for (c, _) in flow],
            target=[tool_idx[t] for (_, t) in flow],
            value=list(flow.values()),
            color="rgba(33,150,243,0.3)",
        ),
    ))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What to notice:** the Sankey shows the agent picking *exactly one* tool per "
        "category — clean separation. In a more complex benchmark you would see multi-tool "
        "chains (e.g. `nasa_events → query_disaster_csv` for a follow-up question)."
    )

    tool_emoji = {
        "query_disaster_csv": "📑",
        "search_nasa_events": "🛰️",
        "search_gdacs_events": "🚨",
        "query_disaster_knowledge": "📚",
        "classify_disaster_image": "🖼️",
    }
    category_emoji = {
        "csv_query":            "📑",
        "nasa_events":          "🛰️",
        "rag_knowledge":        "📚",
        "image_classification": "🖼️",
    }

    st.markdown("### Conversation samples")
    for turn in DATA.AGENT_TURNS:
        tool_str = "  ".join(f"{tool_emoji.get(t, '🔧')} `{t}`" for t in turn["tools_used"])
        cat_em = category_emoji.get(turn["category"], "💬")
        with st.expander(f"{cat_em}  [{turn['category']}]  {turn['question']}"):
            # If this is an image-classification turn, show the actual image being classified
            if turn["category"] == "image_classification":
                # Find which Colombia image is referenced in the question (image_1.jpeg etc.)
                img_name = None
                for cand in ("image_1.jpeg", "image_2.jpeg", "image_3.jpeg"):
                    if cand in turn["question"]:
                        img_name = cand
                        break
                img_name = img_name or "image_1.jpeg"
                img_path = COLOMBIA_IMAGES_DIR / img_name
                ic1, ic2 = st.columns([1, 2])
                with ic1:
                    if img_path.exists():
                        st.image(str(img_path), caption=f"Input → {img_name}",
                                 use_container_width=True)
                    else:
                        st.warning(f"Image not bundled: `{img_name}`")
                with ic2:
                    st.markdown(f"**Tools used:**  {tool_str}")
                    st.markdown(f"**Latency:**  ⏱️ {turn['latency_ms']:,} ms")
                    st.markdown("**Answer:**")
                    st.write(turn["answer"])
            else:
                st.markdown(f"**Tools used:**  {tool_str}")
                st.markdown(f"**Latency:**  ⏱️ {turn['latency_ms']:,} ms")
                st.markdown("**Answer:**")
                st.write(turn["answer"])


# -----------------------------------------------------------------------------
# Page: Image Classification
# -----------------------------------------------------------------------------

def page_classification():
    st.title("🖼️ Image Classification (LLaVA via Ollama)")
    st.caption("30 images evaluated, 5 per class across 6 classes.")

    # ---- Sample images grid ----
    st.markdown("### What the model sees — one sample per class")
    classes_order = ["fire", "flood", "landslide", "earthquake", "smoke", "normal"]
    cols = st.columns(6)
    for col, cls in zip(cols, classes_order):
        img_path = SAMPLE_IMAGES_DIR / f"{cls}.jpg"
        with col:
            if img_path.exists():
                st.image(str(img_path), caption=cls.upper(), use_container_width=True)
            else:
                st.markdown(f"`{cls}` (image not bundled)")
    st.caption(
        "These are downsized samples from the Kaggle 5-class disaster image set "
        "(plus an earthquake set). The full dataset (~9k images) lives in `info_rag/` "
        "and is not bundled in the repo."
    )
    st.divider()

    df = pd.DataFrame(
        DATA.CLASSIFICATION_RESULTS,
        columns=["true", "predicted", "confidence", "correct"],
    )
    n = len(df)
    correct = int(df["correct"].sum())
    acc = correct / n
    z = 1.96
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    margin = z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n) / denom
    ci_low, ci_high = max(0.0, center - margin), min(1.0, center + margin)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Images", f"{n}", "5 per class × 6 classes")
    with c2: metric_card("Accuracy", f"{acc:.0%}", f"95% CI {ci_low:.0%} – {ci_high:.0%}")
    with c3: metric_card("Correct", f"{correct}", f"{n - correct} wrong")
    with c4: metric_card("Classes", "6", "fire/flood/landslide/earthquake/smoke/normal")

    # Confusion matrix
    st.markdown("### Confusion matrix")
    classes = sorted(set(df["true"]) | set(df["predicted"]))
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for _, r in df.iterrows():
        mat[idx[r["true"]]][idx[r["predicted"]]] += 1
    fig = px.imshow(
        mat, x=classes, y=classes, text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="count"),
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What to notice:** the strong diagonal means most classes are predicted correctly. "
        "Off-diagonal cells reveal the model's blind spots — e.g. a *fire ↔ smoke* confusion "
        "is common because they share visual cues, and *landslide ↔ flood* both involve "
        "mud / water textures. These are exactly the failure modes a real deployment would "
        "need to harden against (e.g. by adding a second-stage verifier)."
    )

    # Per-class metrics
    st.markdown("### Per-class precision / recall / F1")
    rows = []
    for cls in classes:
        tp = int(((df["true"] == cls) & (df["predicted"] == cls)).sum())
        fp = int(((df["true"] != cls) & (df["predicted"] == cls)).sum())
        fn = int(((df["true"] == cls) & (df["predicted"] != cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        rows.append({"class": cls, "support": tp + fn,
                     "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)})
    per_class = pd.DataFrame(rows)
    st.dataframe(
        per_class.style
            .background_gradient(subset=["precision", "recall", "f1"], cmap="RdYlGn", vmin=0, vmax=1)
            .format(precision=3),
        use_container_width=True, hide_index=True,
    )

    # Confidence calibration
    st.markdown("### Confidence calibration — does LLaVA's confidence label predict accuracy?")
    cal = (df.groupby("confidence").agg(n=("correct", "size"),
                                        accuracy=("correct", "mean"))
                    .reindex(["high", "medium", "low"]).dropna(subset=["n"]))
    fig = px.bar(
        cal.reset_index(), x="confidence", y="accuracy",
        color="confidence", color_discrete_map={"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"},
        text=cal["accuracy"].map(lambda v: f"{v:.0%}").tolist(),
        title="Accuracy by stated confidence",
    )
    fig.update_layout(height=360, yaxis_range=[0, 1.05], showlegend=False)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**What to notice:** ideally the bars descend left-to-right (high > medium > low). "
        "If they don't, the model's self-reported confidence is decorative — it can't be used "
        "as a filter to auto-route low-confidence cases to a human reviewer. Calibration is "
        "what separates a confidence label that looks useful from one that actually is."
    )


# -----------------------------------------------------------------------------
# Page: Real-World Test (Colombia)
# -----------------------------------------------------------------------------

def page_colombia():
    st.title("🌎 Real-World Test — Colombia Disaster Images")
    st.caption(
        "Out-of-distribution validation: 3 real images from public news media, "
        "not part of any training/evaluation set."
    )

    type_color = {
        "fire":      "#e74c3c",
        "flood":     "#3498db",
        "landslide": "#8d6e63",
        "earthquake": "#f39c12",
        "smoke":     "#7f8c8d",
        "normal":    "#2ecc71",
    }

    for r in DATA.COLOMBIA_RESULTS:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                img_path = COLOMBIA_IMAGES_DIR / r["file"]
                if img_path.exists():
                    st.image(
                        str(img_path),
                        caption=r["file"],
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Image not found: `{r['file']}`")
            with c2:
                badge_color = type_color.get(r["disaster_type"], "#1976d2")
                st.markdown(
                    f"<span class='badge' style='background:{badge_color};color:white;font-size:13px;padding:4px 12px;'>"
                    f"{r['disaster_type'].upper()}</span> "
                    f"<span class='small-caption' style='margin-left:10px;'>"
                    f"confidence: <b>{r['confidence']}</b> · {r['size']} px</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**LLaVA description**")
                st.write(r["description"])


# -----------------------------------------------------------------------------
# Page: Evaluation
# -----------------------------------------------------------------------------

def page_evaluation():
    st.title("📈 Evaluation Dashboard")
    s = DATA.EVALUATION["summary"]

    st.markdown("### Overall metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Composite score", f"{s['avg_composite_score']:.3f}", "weighted (0-1)")
    with c2: metric_card("Pass rate", f"{s['pass_rate']:.0%}", f"{s['total_evaluated']} questions")
    with c3: metric_card("Tool selection", f"{s['tool_selection_accuracy']:.0%}",
                         f"random baseline {DATA.EVALUATION['baseline']['random_tool_accuracy']:.0%}")
    with c4: metric_card("Latency p50 / p95", f"{s['p50_latency_ms']:,} / {s['p95_latency_ms']:,} ms",
                         "per agent turn")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Answer Relevancy", f"{s['avg_answer_relevancy']:.2f}", "1-5 (LLM-judge)")
    with c2: metric_card("Groundedness",     f"{s['avg_factual_groundedness']:.2f}", "1-5 (LLM-judge)")
    with c3: metric_card("Keyword Coverage", f"{s['avg_keyword_coverage']:.0%}", "deterministic")
    with c4:
        cp = s.get("avg_context_precision")
        metric_card("Context Precision", f"{cp:.0%}" if cp else "n/a",
                    f"{s['context_precision_n']} RAG-routed (RAGAS-inspired)")

    # Radar
    st.markdown("### Radar — overall metrics (normalized 0-1)")
    metrics = {
        "Composite": s["avg_composite_score"],
        "Pass rate": s["pass_rate"],
        "Relevancy": s["avg_answer_relevancy"] / 5,
        "Groundedness": s["avg_factual_groundedness"] / 5,
        "Keyword cov.": s["avg_keyword_coverage"],
        "Context prec.": s.get("avg_context_precision") or 0,
        "Tool sel.": s["tool_selection_accuracy"],
    }
    cats = list(metrics.keys()) + [list(metrics.keys())[0]]
    vals = list(metrics.values()) + [list(metrics.values())[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                      line=dict(color="#1976d2"), name="IA_FINAL"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False, height=420,
                      title="Evaluation metrics (normalized)")
    st.plotly_chart(fig, use_container_width=True)

    # Per-category
    st.markdown("### Per-category breakdown")
    cat_rows = []
    for cat, m in DATA.EVALUATION["per_category"].items():
        cat_rows.append({"category": cat, **m})
    cat_df = pd.DataFrame(cat_rows)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cat_df, x="category", y=["avg_composite", "pass_rate"],
                     barmode="group", title="Composite vs. pass rate by category",
                     color_discrete_sequence=["#3498db", "#9b59b6"])
        fig.update_layout(height=380, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(
            cat_df.style
                .background_gradient(subset=["avg_composite", "pass_rate"], cmap="RdYlGn", vmin=0, vmax=1)
                .background_gradient(subset=["avg_relevancy", "avg_groundedness"], cmap="RdYlGn", vmin=1, vmax=5)
                .format({"avg_composite": "{:.2f}", "pass_rate": "{:.0%}",
                         "avg_relevancy": "{:.2f}", "avg_groundedness": "{:.2f}",
                         "avg_keyword_coverage": "{:.0%}"}),
            use_container_width=True, hide_index=True,
        )

    # Baseline comparison
    st.markdown("### Baseline comparison — agent vs. uniform-random tool selection")
    bl = DATA.EVALUATION["baseline"]
    bl_df = pd.DataFrame({
        "method": ["Random baseline", "Agent (ReAct)"],
        "accuracy": [bl["random_tool_accuracy"], bl["agent_tool_accuracy"]],
    })
    fig = px.bar(bl_df, x="method", y="accuracy",
                 color="method", color_discrete_sequence=["#bdc3c7", "#2ecc71"],
                 text=bl_df["accuracy"].map(lambda v: f"{v:.0%}").tolist())
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, yaxis_range=[0, 1.05], showlegend=False,
                      title=f"Agent beats random by {(bl['agent_tool_accuracy'] / bl['random_tool_accuracy']):.1f}x")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Why this matters:** without a baseline, a 92% tool-selection accuracy is just a "
        "number. Comparing against uniform-random tool selection (4 tools → 25% expected) "
        "shows the agent is actually doing useful routing, not getting lucky. This is the "
        "minimum a non-trivial agent must beat."
    )

    # Per-question table
    st.markdown("### Per-question detail")
    detail_df = pd.DataFrame(
        DATA.EVALUATION["details"],
        columns=["category", "question", "relevancy", "groundedness",
                 "keyword_cov", "composite", "passed"],
    )
    detail_df["result"] = detail_df["passed"].map(lambda p: "PASS" if p else "FAIL")
    show = detail_df[["category", "question", "relevancy", "groundedness",
                       "keyword_cov", "composite", "result"]]
    st.dataframe(
        show.style
            .background_gradient(subset=["relevancy", "groundedness"], cmap="RdYlGn", vmin=1, vmax=5)
            .background_gradient(subset=["keyword_cov", "composite"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"relevancy": "{:.0f}", "groundedness": "{:.0f}",
                     "keyword_cov": "{:.0%}", "composite": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )


# -----------------------------------------------------------------------------
# Page: Security
# -----------------------------------------------------------------------------

def page_security():
    st.title("🛡️ Security Considerations")
    st.caption("OWASP Top 10 for LLMs 2025 controls actually enforced in code.")

    sec_df = pd.DataFrame(
        DATA.SECURITY_CONTROLS,
        columns=["OWASP code", "Risk", "Control enforced in this project"],
    )
    st.dataframe(sec_df, use_container_width=True, hide_index=True)

    st.info(
        "**Not claimed:** confidence calibration of the injection detector, exhaustive "
        "jailbreak coverage, PII scrubbing on the indexed CSVs, or a formal threat model. "
        "Detailed risk analysis lives in the `ia-security-task` companion project."
    )


# -----------------------------------------------------------------------------
# Page: Conclusions
# -----------------------------------------------------------------------------

def page_conclusions():
    st.title("📝 Conclusions")
    st.markdown("### What works well")
    for item in DATA.WHAT_WORKS:
        st.markdown(f"- {item}")

    st.markdown("### What I would improve next")
    for item in DATA.WHAT_TO_IMPROVE:
        st.markdown(f"- {item}")

    st.markdown("### Things I learned")
    for item in DATA.LESSONS_LEARNED:
        st.markdown(f"- {item}")

    st.divider()
    st.caption(
        "This dashboard mirrors the analysis carried out in the project notebook. "
        "Reproduce by running `disaster_chatbot.ipynb` end-to-end with Ollama "
        "(llama3.2 + llava) running locally."
    )


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

PAGES = {
    "🏠 Overview": page_overview,
    "📊 Data Exploration": page_data,
    "🤖 Agent Demo": page_agent,
    "🖼️ Image Classification": page_classification,
    "🌎 Real-World Test (Colombia)": page_colombia,
    "📈 Evaluation": page_evaluation,
    "🛡️ Security": page_security,
    "📝 Conclusions": page_conclusions,
}

PAGES[page]()
