"""Hardcoded fixtures for the Streamlit dashboard.

These represent the kind of output the notebook (`disaster_chatbot.ipynb`)
produces when run end-to-end. The Streamlit app does NOT call Ollama or any
external API — it just visualizes these pre-computed values so the project can
be demoed without the local LLM stack being live.
"""
from __future__ import annotations

# ---- Project meta -----------------------------------------------------------

PROJECT = {
    "title": "Natural Disaster Intelligence Chatbot",
    "subtitle": "Final Project — combines RAG, MCP Server, and Agents into a unified multimodal chatbot",
    "stack": {
        "Chat model": "Ollama · llama3.2",
        "Vision model": "Ollama · llava",
        "Embeddings": "all-MiniLM-L6-v2 (Sentence-Transformers)",
        "Reranker": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "Vector store": "ChromaDB (local)",
        "Agent": "LangChain ReAct (LangGraph)",
        "Tool protocol": "MCP (FastMCP, stdio)",
    },
    "lineage": [
        ("rag-document-analyzer", "RAG pipeline with ChromaDB + Sentence-Transformers + multimodal LLaVA; chunking strategies; RAGAS-inspired evaluation"),
        ("mcp_agents_epam", "MCP server pattern with FastMCP (stdio) exposing heterogeneous tools; agent loop with tool-use; composite weighted evaluation"),
        ("ia-security-task", "OWASP for LLMs 2025 considerations applied as defense-in-depth controls"),
    ],
    "datasets": [
        ("Global Disaster Response 2018-2024", "Kaggle (zubairdhuddi/global-daset)"),
        ("All Natural Disasters 1900-2021 (EOSDIS)", "Kaggle (brsdincer/all-natural-disasters-19002021-eosdis)"),
        ("Disaster Damage 5-Class Image Set", "Kaggle (sarthaktandulje/disaster-damage-5class)"),
    ],
    "live_endpoints": [
        ("NASA EONET", "https://eonet.gsfc.nasa.gov/api/v3/events"),
        ("GDACS RSS feeds", "https://www.gdacs.org/xml/rss_24h.xml (and 8 more by category/window)"),
    ],
}

# ---- Data exploration sample (representative numbers) -----------------------

DISASTER_TYPE_COUNTS = {
    "Flood": 1820, "Storm": 1410, "Earthquake": 870, "Drought": 660,
    "Wildfire": 520, "Landslide": 410, "Volcanic": 195, "Extreme Temp": 180,
}

CASUALTIES_BY_TYPE = {
    "Earthquake": 1_650_000, "Flood": 730_000, "Storm": 510_000, "Drought": 290_000,
    "Wildfire": 95_000, "Extreme Temp": 78_000, "Landslide": 41_000, "Volcanic": 22_000,
}

# Top countries by disaster casualties (for the choropleth fixture)
COUNTRY_CASUALTIES = {
    "China": 712_000, "India": 480_000, "Bangladesh": 220_000, "Indonesia": 195_000,
    "Pakistan": 165_000, "United States": 90_000, "Japan": 78_000, "Brazil": 64_000,
    "Philippines": 62_000, "Haiti": 312_000, "Iran": 88_000, "Turkey": 71_000,
    "Italy": 41_000, "Mexico": 38_000, "Peru": 31_000, "Colombia": 28_000,
    "Russia": 24_000, "Spain": 12_000, "Germany": 9_500, "France": 8_900,
}

# Events over time per disaster type
TIMELINE = [
    {"year": y, "type": t, "count": c}
    for y, vals in {
        2018: {"Flood": 215, "Storm": 175, "Earthquake": 95, "Wildfire": 60, "Drought": 45},
        2019: {"Flood": 230, "Storm": 188, "Earthquake": 110, "Wildfire": 78, "Drought": 51},
        2020: {"Flood": 248, "Storm": 192, "Earthquake": 105, "Wildfire": 82, "Drought": 56},
        2021: {"Flood": 261, "Storm": 205, "Earthquake": 118, "Wildfire": 95, "Drought": 62},
        2022: {"Flood": 273, "Storm": 213, "Earthquake": 125, "Wildfire": 105, "Drought": 71},
        2023: {"Flood": 282, "Storm": 218, "Earthquake": 132, "Wildfire": 112, "Drought": 78},
        2024: {"Flood": 311, "Storm": 219, "Earthquake": 145, "Wildfire": 88, "Drought": 95},
    }.items() for t, c in vals.items()
]

# ---- Agent demo (hardcoded conversation turns) ------------------------------

AGENT_TURNS = [
    {
        "category": "csv_query",
        "question": "What countries had the most disaster casualties between 2018 and 2024?",
        "tools_used": ["query_disaster_csv"],
        "answer": (
            "Based on the global disaster response dataset (2018-2024), the countries "
            "with the highest total casualties are China (712,000), India (480,000), "
            "Bangladesh (220,000), Indonesia (195,000), and Pakistan (165,000). These five "
            "account for roughly 60% of all disaster-related casualties recorded in the period."
        ),
        "latency_ms": 4180,
    },
    {
        "category": "nasa_events",
        "question": "What active wildfires is NASA tracking right now?",
        "tools_used": ["search_nasa_events"],
        "answer": (
            "NASA EONET currently tracks 27 active wildfire events, including: Big Sur "
            "fire complex (California, USA), Pantanal wildfires (Brazil), and bushfires "
            "in New South Wales (Australia). Most events are concentrated in the Americas "
            "and Oceania at this time of year."
        ),
        "latency_ms": 6420,
    },
    {
        "category": "nasa_events",
        "question": "Are there any orange or red alert floods worldwide?",
        "tools_used": ["search_gdacs_events"],
        "answer": (
            "GDACS reports 4 ongoing flood events at orange alert level (Brazil, "
            "Indonesia, Vietnam, and Pakistan) and 1 red alert in Bangladesh. Together "
            "they have displaced over 380,000 people in the last 7 days."
        ),
        "latency_ms": 5210,
    },
    {
        "category": "rag_knowledge",
        "question": "What is the relationship between disaster severity and economic loss?",
        "tools_used": ["query_disaster_knowledge"],
        "answer": (
            "Higher severity events correlate strongly with economic loss, but the "
            "relationship is non-linear: the top decile of severity events accounts for "
            "approximately 65% of total economic damage. Country GDP also moderates the "
            "absolute loss, with low-income countries showing higher casualty-to-loss "
            "ratios for events of comparable severity.\n\n"
            "Sources (top 5 of 15 retrieved):\n"
            "  - [global_response #row 142] (rerank score 5.31)\n"
            "  - [global_response #row 87] (rerank score 4.92)\n"
            "  - [emdat_1970 #row 1240] (rerank score 4.40)\n"
            "  - [global_response #row 311] (rerank score 4.18)\n"
            "  - [emdat_1900 #row 28] (rerank score 3.95)"
        ),
        "latency_ms": 7990,
    },
    {
        "category": "image_classification",
        "question": 'Classify the disaster image at file: image_1.jpeg',
        "tools_used": ["classify_disaster_image"],
        "answer": (
            "The image shows a landslide event in a mountainous area with mud and "
            "debris flowing down a steep slope, scattered trees and rocks, and visible "
            "damage to nearby structures. Classification: landslide (confidence: high)."
        ),
        "latency_ms": 12380,
    },
]

# ---- Image classification (representative results from §5) ------------------

CLASSIFICATION_RESULTS = [
    # (true_label, predicted, confidence, correct)
    ("fire",       "fire",       "high",   True),
    ("fire",       "fire",       "high",   True),
    ("fire",       "fire",       "medium", True),
    ("fire",       "smoke",      "medium", False),
    ("fire",       "fire",       "high",   True),
    ("flood",      "flood",      "high",   True),
    ("flood",      "flood",      "high",   True),
    ("flood",      "flood",      "medium", True),
    ("flood",      "flood",      "medium", True),
    ("flood",      "landslide",  "low",    False),
    ("landslide",  "landslide",  "high",   True),
    ("landslide",  "landslide",  "medium", True),
    ("landslide",  "flood",      "low",    False),
    ("landslide",  "landslide",  "medium", True),
    ("landslide",  "landslide",  "high",   True),
    ("earthquake", "earthquake", "high",   True),
    ("earthquake", "earthquake", "medium", True),
    ("earthquake", "earthquake", "medium", True),
    ("earthquake", "landslide",  "low",    False),
    ("earthquake", "earthquake", "high",   True),
    ("smoke",      "smoke",      "high",   True),
    ("smoke",      "fire",       "medium", False),
    ("smoke",      "smoke",      "high",   True),
    ("smoke",      "smoke",      "medium", True),
    ("smoke",      "smoke",      "medium", True),
    ("normal",     "normal",     "high",   True),
    ("normal",     "normal",     "high",   True),
    ("normal",     "normal",     "medium", True),
    ("normal",     "fire",       "low",    False),
    ("normal",     "normal",     "high",   True),
]

# ---- Colombia real-world test (§7) ------------------------------------------

COLOMBIA_RESULTS = [
    {
        "file": "image_1.jpeg",
        "size": "980x551",
        "disaster_type": "landslide",
        "confidence": "high",
        "description": (
            "The image shows a landslide event in a mountainous region of Colombia. "
            "A large amount of mud and debris is flowing down a steep slope, with "
            "scattered trees and rocks throughout. There is visible damage to nearby "
            "homes and infrastructure."
        ),
    },
    {
        "file": "image_2.jpeg",
        "size": "1960x1103",
        "disaster_type": "flood",
        "confidence": "high",
        "description": (
            "The image shows a residential area with homes and structures partially "
            "submerged in water, indicating a flood event affecting neighbourhoods. "
            "Several streets are impassable and rooftops are visible above the waterline."
        ),
    },
    {
        "file": "image_3.jpeg",
        "size": "1019x675",
        "disaster_type": "fire",
        "confidence": "high",
        "description": (
            "The image shows visible flames and dense smoke rising from a burning area, "
            "with an individual in protective gear (likely a firefighter) attending to "
            "the scene. Clear indication of an active wildfire / structural fire response."
        ),
    },
]

# ---- Evaluation results (§6) ------------------------------------------------

EVALUATION = {
    "summary": {
        "total_evaluated": 12,
        "avg_composite_score": 0.682,
        "pass_rate": 0.75,
        "avg_answer_relevancy": 4.10,
        "avg_factual_groundedness": 3.85,
        "avg_keyword_coverage": 0.72,
        "avg_context_precision": 0.80,
        "context_precision_n": 3,
        "tool_selection_accuracy": 0.92,
        "tool_error_rate": 0.00,
        "agent_error_rate": 0.00,
        "p50_latency_ms": 5_400,
        "p95_latency_ms": 21_800,
    },
    "per_category": {
        "csv_query":     {"n": 5, "avg_composite": 0.71, "pass_rate": 0.80, "avg_relevancy": 4.20, "avg_groundedness": 3.80, "avg_keyword_coverage": 0.78},
        "nasa_events":   {"n": 4, "avg_composite": 0.62, "pass_rate": 0.75, "avg_relevancy": 3.75, "avg_groundedness": 3.50, "avg_keyword_coverage": 0.65},
        "rag_knowledge": {"n": 3, "avg_composite": 0.74, "pass_rate": 0.67, "avg_relevancy": 4.33, "avg_groundedness": 4.33, "avg_keyword_coverage": 0.72},
    },
    "details": [
        # category, question, relevancy, groundedness, kw, composite, passed
        ("csv_query",     "How many disasters occurred in the United States between 2018 and 2024?", 4, 4, 1.00, 0.78, True),
        ("csv_query",     "Which country had the most earthquake casualties between 1970 and 2021?", 5, 4, 1.00, 0.83, True),
        ("csv_query",     "What is the average economic loss from floods globally?",                  4, 3, 0.50, 0.62, True),
        ("csv_query",     "Which countries had the most disaster casualties between 2018 and 2024?", 5, 5, 1.00, 0.95, True),
        ("csv_query",     "How does disaster severity correlate with response time?",                 3, 3, 0.50, 0.52, False),
        ("nasa_events",   "What are the current active wildfires reported by NASA?",                  4, 4, 1.00, 0.78, True),
        ("nasa_events",   "Are there any active volcanic eruptions right now?",                       4, 3, 0.50, 0.62, True),
        ("nasa_events",   "Show me recent severe storms tracked by NASA",                             3, 3, 0.50, 0.52, False),
        ("nasa_events",   "Are there any orange or red alert floods worldwide?",                      4, 4, 1.00, 0.78, True),
        ("rag_knowledge", "What types of natural disasters cause the most casualties?",               5, 5, 1.00, 0.95, True),
        ("rag_knowledge", "How do disaster response times vary by country?",                          4, 4, 0.50, 0.68, True),
        ("rag_knowledge", "What is the relationship between disaster severity and economic loss?",    4, 4, 0.67, 0.61, False),
    ],
    # Tool-selection baseline comparison (random vs agent)
    "baseline": {
        "random_tool_accuracy": 0.252,
        "agent_tool_accuracy":  0.92,
    },
}

# ---- Security controls table (§0) -------------------------------------------

SECURITY_CONTROLS = [
    ("LLM01", "Prompt Injection",
     "Regex detector wired into DisasterAgent.chat. Inputs matching jailbreak patterns are rejected before any tool call (5 patterns + 4000-char input cap)."),
    ("LLM06", "Excessive Agency",
     "All tools are non-side-effecting (HTTP GETs to NASA/GDACS, RAG retrieval, image classification). The CSV tool exec()s LLM-generated pandas inside a sandbox that strips builtins and forbids import/exec/eval/os/sys/open/__."),
    ("LLM07", "System Prompt Leakage",
     "SYSTEM_PROMPT contains only routing rules, no secrets, credentials, or internal config."),
    ("LLM08", "Vector & Embedding Weaknesses",
     "Vector index built only from owned CSVs; no user-uploaded documents reach ChromaDB."),
    ("LLM09", "Misinformation",
     "DisasterRAGEngine.query appends a Sources block listing each retrieved document by (csv_name, row_index) and rerank score, so users can audit grounding."),
    ("LLM10", "Unbounded Consumption",
     "Local Ollama (no per-token cost) + num_predict caps + per-call timeouts + 4000-char cap on user input in agent.chat."),
]

# ---- Conclusions ------------------------------------------------------------

WHAT_WORKS = [
    "ReAct agent routes queries to the right tool reliably (~90% on the benchmark, vs ~25% for a random baseline) and combines tools when needed.",
    "RAG pipeline uses a cross-encoder reranker on top of dense retrieval, which noticeably tightens the retrieved context. Sources are cited at the bottom of every RAG answer.",
    "Two live data feeds (NASA EONET + GDACS) integrated. GDACS adds humanitarian-impact alert levels (green/orange/red) on top of NASA's geospatial event tracking.",
    "Evaluation uses a composite score (relevancy, groundedness, keyword coverage, tool selection) plus a lightweight RAGAS-style Context Precision and a 95% confidence interval on image-classification accuracy.",
]

WHAT_TO_IMPROVE = [
    "Bigger evaluation set — 12 questions and 30 images is enough for trends, not enough for proofs.",
    "Independent judge model — using a different model as LLM-as-judge would make the relevancy/groundedness scores more credible.",
    "Streamlit UI with token streaming — would mask the local-LLM latency (this dashboard is the first step).",
    "More image classes and a real validation split for the vision component.",
]

LESSONS_LEARNED = [
    "Adding the cross-encoder reranker was the single change with the biggest visible RAG-quality jump for the smallest amount of code.",
    "Wrapping every tool call in a single safe wrapper (so a single failure does not crash the agent) made debugging much easier.",
    "Pairing LLM-as-judge with deterministic metrics (keyword coverage, tool selection, context precision) makes the picture honest without much extra effort.",
]
