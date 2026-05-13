import base64
import json
import logging
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Silence transformers + every known sub-logger that emits non-actionable
# warnings on small CPU-only model loads (the most stubborn one is
# `transformers.integrations.tensor_parallel`, which logs a multi-line
# "The following layers were not sharded" warning at WARNING level).
for noisy in (
    "transformers",
    "transformers.modeling_utils",
    "transformers.integrations.tensor_parallel",
    "transformers.configuration_utils",
    "huggingface_hub",
):
    _lg = logging.getLogger(noisy)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except ImportError:
    pass

import time
from functools import lru_cache

import requests
import chromadb
import pandas as pd
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL, MODELS, EMBEDDING_CONFIG, CHROMA_CONFIG,
    CSV_FILES, DISASTER_IMAGES_DIR, EARTHQUAKE_IMAGES_DIR,
    DISASTER_CATEGORIES, VECTOR_DB_DIR,
    LLM_OUTPUT_LIMITS, HTTP_RETRY, RETRIEVAL_CONFIG,
)

logger = logging.getLogger(__name__)


def _ollama_post_with_retry(payload: dict, timeout: int) -> str:
    """POST to Ollama /api/generate with retry + backoff. Returns the response string."""
    last_exc = None
    for attempt in range(1, HTTP_RETRY["attempts"] + 1):
        try:
            r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except requests.RequestException as e:
            last_exc = e
            sleep = HTTP_RETRY["backoff_seconds"] * (2 ** (attempt - 1))
            logger.warning("ollama POST attempt=%d/%d failed: %s — backoff %.1fs",
                           attempt, HTTP_RETRY["attempts"], e, sleep)
            if attempt < HTTP_RETRY["attempts"]:
                time.sleep(sleep)
    raise last_exc


class DisasterRAGEngine:
    def __init__(self):
        logger.info("Loading embedding model: %s", EMBEDDING_CONFIG["model"])
        self.embedding_model = SentenceTransformer(EMBEDDING_CONFIG["model"])
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_CONFIG["persist_directory"])
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_CONFIG["collection_name"],
            metadata={"hnsw:space": CHROMA_CONFIG["distance_metric"]},
        )
        self._indexed = self.collection.count() > 0
        # Lazy-loaded cross-encoder for re-ranking (loaded on first query call).
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder reranker: %s", RETRIEVAL_CONFIG["rerank_model"])
            self._reranker = CrossEncoder(RETRIEVAL_CONFIG["rerank_model"])
        return self._reranker

    @lru_cache(maxsize=512)
    def _embed_cached(self, text: str) -> tuple:
        """LRU-cached single-string embedding. Returns a tuple so it's hashable."""
        return tuple(self.embedding_model.encode(text).tolist())

    def index_disaster_data(self, force: bool = False):
        if self._indexed and not force:
            return f"Already indexed {self.collection.count()} documents. Use force=True to re-index."

        if force:
            self.chroma_client.delete_collection(CHROMA_CONFIG["collection_name"])
            self.collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_CONFIG["collection_name"],
                metadata={"hnsw:space": CHROMA_CONFIG["distance_metric"]},
            )

        documents = []
        metadatas = []
        ids = []

        for csv_name, csv_path in CSV_FILES.items():
            if not Path(csv_path).exists():
                continue
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = [c.strip().lower().replace(" ", "_").replace("'", "") for c in df.columns]

            for idx, row in df.iterrows():
                if idx >= 5000:
                    break
                doc = self._row_to_document(row, csv_name)
                if doc:
                    documents.append(doc)
                    metadatas.append({"source": csv_name, "row_index": int(idx)})
                    ids.append(f"{csv_name}_{idx}")

        if not documents:
            return "No documents to index."

        batch_size = EMBEDDING_CONFIG["batch_size"]
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch_docs).tolist()
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        self._indexed = True
        return f"Indexed {len(documents)} documents into ChromaDB."

    def _row_to_document(self, row, source: str) -> Optional[str]:
        parts = []
        if source == "global_response":
            for field in ["date", "country", "disaster_type", "severity_index",
                          "casualties", "economic_loss_usd", "response_time_hours"]:
                if field in row.index and pd.notna(row.get(field)):
                    parts.append(f"{field}: {row[field]}")
        else:
            for field in ["year", "country", "disaster_type", "disaster_subtype",
                          "total_deaths", "total_affected", "total_damages_(000_us$)"]:
                col = field.replace("(", "").replace(")", "").replace("$", "").replace("'", "")
                for c in row.index:
                    if col.replace("_", "") in c.replace("_", ""):
                        if pd.notna(row[c]):
                            parts.append(f"{c}: {row[c]}")
                        break
        return " | ".join(parts) if parts else None

    def retrieve_only(self, question: str, n_results: Optional[int] = None) -> list:
        """Return retrieved + re-ranked chunks WITHOUT calling the LLM.

        Used by the evaluator (`evaluator.py`) to compute Context Precision —
        a RAGAS-style metric that measures retrieval quality independently of
        generation quality. Returns a list of dicts:
            [{"text": ..., "metadata": {...}, "rerank_score": float}, ...]
        Empty list if the index is empty or no candidates pass the threshold.
        """
        if not self._indexed:
            return []
        candidate_k = RETRIEVAL_CONFIG["candidate_k"]
        final_n = n_results or RETRIEVAL_CONFIG["final_n"]

        query_embedding = list(self._embed_cached(question))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
        )
        candidates = results["documents"][0] if results["documents"] else []
        cand_meta = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [(question, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        triples = list(zip(scores, candidates, cand_meta or [{}] * len(candidates)))
        ranked = sorted(triples, key=lambda x: -x[0])
        kept = [(s, d, m) for s, d, m in ranked if s >= RETRIEVAL_CONFIG["min_score"]][:final_n]
        return [{"text": d, "metadata": m if isinstance(m, dict) else {},
                 "rerank_score": float(s)} for s, d, m in kept]

    def query(self, question: str, n_results: Optional[int] = None) -> str:
        """Retrieve top candidate_k via embeddings, re-rank with cross-encoder, keep final_n.

        Returns the LLM answer followed by a Sources section citing each retrieved
        document by its (source_csv, row_index) — supports OWASP LLM09 (Misinformation):
        the user can audit which records grounded the answer.
        """
        if not self._indexed:
            return "Knowledge base not indexed yet. Call index_disaster_data() first."

        candidate_k = RETRIEVAL_CONFIG["candidate_k"]
        final_n = n_results or RETRIEVAL_CONFIG["final_n"]
        t0 = time.time()

        query_embedding = list(self._embed_cached(question))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
        )
        candidates = results["documents"][0] if results["documents"] else []
        cand_meta = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        if not candidates:
            return "No relevant documents found."

        # Cross-encoder re-rank: score each (question, candidate) pair.
        reranker = self._get_reranker()
        pairs = [(question, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        triples = list(zip(scores, candidates, cand_meta or [{}] * len(candidates)))
        ranked = sorted(triples, key=lambda x: -x[0])
        kept = [(s, d, m) for s, d, m in ranked if s >= RETRIEVAL_CONFIG["min_score"]][:final_n]

        logger.info(
            "rag.query candidate_k=%d kept=%d top_score=%.3f latency_ms=%d",
            len(candidates), len(kept), float(kept[0][0]) if kept else 0.0,
            int((time.time() - t0) * 1000),
        )

        if not kept:
            return "No documents passed the re-ranking threshold."

        context = "\n".join(d for _, d, _ in kept)
        answer = self._generate_answer(question, context)

        # OWASP LLM09 — append source citations so the user can verify.
        sources = []
        for s, _, m in kept:
            src = m.get("source", "unknown") if isinstance(m, dict) else "unknown"
            row = m.get("row_index", "?") if isinstance(m, dict) else "?"
            sources.append(f"  - [{src} #row {row}] (rerank score {float(s):.2f})")
        return f"{answer}\n\nSources (top {len(kept)} of {len(candidates)} retrieved):\n" + "\n".join(sources)

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""Based on the following disaster data context, answer the question accurately.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        return _ollama_post_with_retry(
            payload={
                "model": MODELS["chat"]["name"],
                "prompt": prompt,
                "stream": False,
                "temperature": MODELS["chat"]["temperature"],
                "options": {"num_predict": LLM_OUTPUT_LIMITS["chat_num_predict"]},
            },
            timeout=HTTP_RETRY["request_timeout"],
        )

    def classify_image(self, image_path: str) -> dict:
        path = Path(image_path)
        if not path.exists():
            return {"error": f"Image not found: {image_path}"}

        image_b64 = self._image_to_base64(path)

        prompt = (
            f"Analyze this image and classify it as one of these disaster types: "
            f"{', '.join(DISASTER_CATEGORIES)}. "
            f"Respond with ONLY a JSON object: "
            f'{{"disaster_type": "<type>", "confidence": "<high/medium/low>", "description": "<brief description>"}}'
        )

        t0 = time.time()
        try:
            raw = _ollama_post_with_retry(
                payload={
                    "model": MODELS["vision"]["name"],
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "temperature": MODELS["vision"]["temperature"],
                    "options": {"num_predict": LLM_OUTPUT_LIMITS["vision_num_predict"]},
                },
                timeout=HTTP_RETRY["vision_timeout"],
            )
            parsed = self._parse_classification(raw)
            logger.info("classify_image %s -> %s (latency_ms=%d)",
                        path.name, parsed.get("disaster_type"), int((time.time() - t0) * 1000))
            return parsed
        except Exception as e:
            logger.exception("classify_image failed for %s: %s", path.name, e)
            return {"error": str(e), "raw_response": ""}

    def _parse_classification(self, raw: str) -> dict:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        raw_lower = raw.lower()
        for cat in DISASTER_CATEGORIES:
            if cat in raw_lower:
                return {"disaster_type": cat, "confidence": "medium", "description": raw.strip()}
        return {"disaster_type": "unknown", "confidence": "low", "description": raw.strip()}

    def _image_to_base64(self, image_path: Path) -> str:
        img = Image.open(image_path)
        max_size = 512
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_sample_images(self, category: str, n: int = 3) -> list:
        if category == "earthquake":
            img_dir = EARTHQUAKE_IMAGES_DIR
        else:
            img_dir = DISASTER_IMAGES_DIR / category
        if not img_dir.exists():
            return []
        images = sorted(img_dir.glob("*.jpg"))[:n]
        return [str(p) for p in images]
