from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "info_rag"
IMAGES_DIR = DATA_DIR / "images"
DISASTER_IMAGES_DIR = IMAGES_DIR / "disaster_dataset" / "disaster_dataset"
EARTHQUAKE_IMAGES_DIR = IMAGES_DIR / "earthquake" / "earthquake"
CSV_DIR = DATA_DIR / "DISASTERS"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EVAL_DATA_DIR = BASE_DIR / "eval_data"

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = {
    "chat": {
        "name": "llama3.2",
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "vision": {
        "name": "llava",
        "temperature": 0.3,
        "top_p": 0.9,
    },
}

EMBEDDING_CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "batch_size": 32,
}

CHROMA_CONFIG = {
    "collection_name": "disaster_knowledge",
    "distance_metric": "cosine",
    "persist_directory": str(VECTOR_DB_DIR),
}

CSV_FILES = {
    "global_response": DATA_DIR / "global_disaster_response_2018_2024 (1).csv",
    "emdat_1970": CSV_DIR / "1970-2021_DISASTERS.xlsx - emdat data.csv",
    "emdat_1900": CSV_DIR / "1900_2021_DISASTERS.xlsx - emdat data.csv",
}

DISASTER_CATEGORIES = ["fire", "flood", "landslide", "earthquake", "smoke", "normal"]

NASA_EONET_BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"
GDACS_BASE_URL = "https://www.gdacs.org/xml"

AGENT_CONFIG = {
    "max_iterations": 10,
    "return_intermediate_steps": True,
}

# --- Reproducibility -------------------------------------------------------
# Seed used for any stochastic component (numpy, random, embedding sampling).
RANDOM_SEED = 42

# Bound LLM output to keep cost / latency predictable.
LLM_OUTPUT_LIMITS = {
    "chat_num_predict": 512,
    "vision_num_predict": 256,
    "judge_num_predict": 256,
}

# Retry policy for any HTTP-backed call (Ollama / NASA EONET).
HTTP_RETRY = {
    "attempts": 3,
    "backoff_seconds": 1.5,
    "request_timeout": 60,
    "vision_timeout": 120,
}

# RAG retrieval + re-ranking knobs.
RETRIEVAL_CONFIG = {
    "candidate_k": 15,           # initial vector search
    "final_n": 5,                # after cross-encoder re-rank
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "min_score": 0.0,            # below this, drop
}

# Evaluation composite-score weights (sum to 1.0).
# Inspired by mcp_agents_epam: a single "pass/fail" metric per question
# combining all four sub-metrics on the same 0-1 scale.
EVAL_WEIGHTS = {
    "relevancy": 0.35,
    "groundedness": 0.25,
    "keyword_coverage": 0.20,
    "tool_selection": 0.20,
}
DEFAULT_PASS_THRESHOLD = 0.60   # composite >= this -> "passed"


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed all relevant libraries for reproducible runs."""
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
