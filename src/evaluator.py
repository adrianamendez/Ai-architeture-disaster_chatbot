import json
import logging
import unicodedata
from collections import defaultdict

import requests
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL, MODELS, EVAL_DATA_DIR, LLM_OUTPUT_LIMITS, HTTP_RETRY,
    EVAL_WEIGHTS, DEFAULT_PASS_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def keyword_coverage(answer: str, keywords: list) -> float:
    """Fraction of expected keywords found in the answer (case + accent insensitive)."""
    if not keywords:
        return 1.0  # no requirement -> full credit
    norm_answer = _strip_accents(answer.lower())
    hits = sum(1 for kw in keywords if _strip_accents(kw.lower()) in norm_answer)
    return hits / len(keywords)


def composite_score(relevancy_1to5: float, groundedness_1to5: float,
                    keyword_cov_0to1: float, tool_correct_bool: bool) -> float:
    """Weighted composite on a 0-1 scale (mcp_agents_epam style)."""
    w = EVAL_WEIGHTS
    return (
        w["relevancy"] * (relevancy_1to5 / 5.0)
        + w["groundedness"] * (groundedness_1to5 / 5.0)
        + w["keyword_coverage"] * keyword_cov_0to1
        + w["tool_selection"] * (1.0 if tool_correct_bool else 0.0)
    )


def context_precision(retrieved_chunks: list, keywords: list) -> float:
    """RAGAS-style Context Precision (lightweight, deterministic variant).

    Definition: fraction of retrieved chunks that contain at least one of the
    expected keywords for the question. A high value means retrieval is on-topic;
    a low value means the reranker surfaced irrelevant chunks.

    Note: the canonical RAGAS definition uses LLM-as-judge per chunk for
    relevance. We use the cheaper keyword-overlap proxy to stay within the
    Ollama budget; both correlate with retrieval quality on short corpora.
    Return 1.0 (no penalty) when the question has no keywords to check against.
    """
    if not retrieved_chunks or not keywords:
        return 1.0
    norm_keywords = [_strip_accents(k.lower()) for k in keywords]
    relevant = 0
    for chunk in retrieved_chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        norm_text = _strip_accents(text.lower())
        if any(kw in norm_text for kw in norm_keywords):
            relevant += 1
    return relevant / len(retrieved_chunks)


def _is_tool_error(text: str) -> bool:
    """Heuristic: detect when a tool call returned an error string instead of a real result."""
    if not isinstance(text, str):
        return False
    markers = [
        "Error processing", "Error executing", "API error",
        "Connection error", "tool '", "[tool ", "failed:",
        "not indexed yet",
    ]
    return any(m in text for m in markers)


class DisasterChatbotEvaluator:
    def __init__(self, eval_dataset_path: Optional[str] = None):
        self.eval_dataset_path = eval_dataset_path or str(EVAL_DATA_DIR / "eval_dataset.json")
        self.eval_data = self._load_eval_data()

    def _load_eval_data(self) -> list:
        path = Path(self.eval_dataset_path)
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    def _llm_judge(self, question: str, answer: str, criterion: str, reference: str = "") -> dict:
        ref_section = f"\nReference answer: {reference}" if reference else ""
        prompt = f"""You are an evaluation judge. Score the following answer on a scale of 1-5.

Criterion: {criterion}

Question: {question}
Answer: {answer}{ref_section}

Scoring guide:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat relevant and partially correct
4 = Mostly correct and relevant
5 = Fully correct, relevant, and comprehensive

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODELS["chat"]["name"],
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "options": {"num_predict": LLM_OUTPUT_LIMITS["judge_num_predict"]},
                },
                timeout=HTTP_RETRY["request_timeout"],
            )
            response.raise_for_status()
            raw = response.json()["response"]
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
        return {"score": 0, "reasoning": "Failed to evaluate"}

    def evaluate_answer_relevancy(self, question: str, answer: str) -> dict:
        return self._llm_judge(
            question, answer,
            "Answer Relevancy: Does the answer directly address the question asked?"
        )

    def evaluate_factual_groundedness(self, question: str, answer: str, context: str) -> dict:
        return self._llm_judge(
            question, answer,
            f"Factual Groundedness: Is the answer supported by the provided context?\nContext: {context}"
        )

    def evaluate_tool_selection(self, question: str, tools_used: list, expected_tool: str) -> dict:
        correct = expected_tool in tools_used
        return {
            "score": 5 if correct else 1,
            "correct": correct,
            "expected": expected_tool,
            "actual": tools_used,
            "reasoning": f"Expected {expected_tool}, got {tools_used}",
        }

    def run_evaluation(self, agent) -> dict:
        if not self.eval_data:
            return {"error": "No evaluation dataset loaded"}

        results = {
            "total": len(self.eval_data),
            "answer_relevancy": [],
            "tool_selection": [],
            "factual_groundedness": [],
            "keyword_coverage": [],
            "context_precision": [],   # RAGAS-style; only populated for RAG-routed questions
            "composite_scores": [],
            "passed_flags": [],
            "details": [],
            "tool_errors": 0,
            "agent_errors": 0,
            "latencies_ms": [],
        }

        # Side-channel access to the RAG engine to inspect retrieved chunks
        # without re-running the LLM. The agent owns the engine.
        rag_engine = getattr(agent, "rag_engine", None)

        for item in self.eval_data:
            question = item["question"]
            expected_tool = item.get("expected_tool", "")
            reference = item.get("reference_answer", "")
            keywords = item.get("keywords", [])
            min_required = item.get("min_composite_score", DEFAULT_PASS_THRESHOLD)
            category = item.get("category", "other")

            agent_result = agent.chat(question)
            answer = agent_result["answer"]
            tools_used = agent_result["tools_used"]
            latency = agent_result.get("latency_ms")
            if latency is not None:
                results["latencies_ms"].append(latency)

            if _is_tool_error(answer):
                results["tool_errors"] += 1
            if answer.startswith("Error processing your question"):
                results["agent_errors"] += 1

            rel = self.evaluate_answer_relevancy(question, answer).get("score", 0)
            ground = self.evaluate_factual_groundedness(question, answer, reference).get("score", 0) if reference else 0
            kw_cov = keyword_coverage(answer, keywords)
            tool_correct = (expected_tool in tools_used) if expected_tool else False

            # Context Precision (RAGAS-style) — only meaningful when we hit RAG.
            ctx_prec = None
            is_rag_query = (expected_tool == "query_disaster_knowledge"
                            or "query_disaster_knowledge" in tools_used)
            if is_rag_query and rag_engine is not None:
                try:
                    chunks = rag_engine.retrieve_only(question)
                    if chunks:
                        ctx_prec = context_precision(chunks, keywords)
                        results["context_precision"].append(ctx_prec)
                except Exception as e:
                    logger.warning("context_precision computation failed: %s", e)

            comp = composite_score(rel, ground, kw_cov, tool_correct)
            passed = comp >= min_required

            results["answer_relevancy"].append(rel)
            results["factual_groundedness"].append(ground)
            results["keyword_coverage"].append(kw_cov)
            results["composite_scores"].append(comp)
            results["passed_flags"].append(passed)
            if expected_tool:
                results["tool_selection"].append(self.evaluate_tool_selection(question, tools_used, expected_tool))

            results["details"].append({
                "question": question,
                "category": category,
                "answer": answer,
                "tools_used": tools_used,
                "relevancy_score": rel,
                "groundedness_score": ground,
                "keyword_coverage": round(kw_cov, 3),
                "context_precision": round(ctx_prec, 3) if ctx_prec is not None else None,
                "composite_score": round(comp, 3),
                "min_required": min_required,
                "passed": passed,
                "latency_ms": latency,
                "had_error": _is_tool_error(answer),
            })

        rel_scores = [s for s in results["answer_relevancy"] if s > 0]
        ground_scores = [s for s in results["factual_groundedness"] if s > 0]
        tool_scores = results["tool_selection"]

        latencies = results["latencies_ms"]
        kw_scores = results["keyword_coverage"]
        comp_scores = results["composite_scores"]
        passed = results["passed_flags"]

        ctx_prec_scores = results["context_precision"]
        results["summary"] = {
            "avg_answer_relevancy": sum(rel_scores) / len(rel_scores) if rel_scores else 0,
            "avg_factual_groundedness": sum(ground_scores) / len(ground_scores) if ground_scores else 0,
            "avg_keyword_coverage": sum(kw_scores) / len(kw_scores) if kw_scores else 0,
            "avg_context_precision": sum(ctx_prec_scores) / len(ctx_prec_scores) if ctx_prec_scores else None,
            "context_precision_n": len(ctx_prec_scores),
            "avg_composite_score": sum(comp_scores) / len(comp_scores) if comp_scores else 0,
            "tool_selection_accuracy": (
                sum(1 for t in tool_scores if t["correct"]) / len(tool_scores)
                if tool_scores else 0
            ),
            "pass_rate": sum(1 for p in passed if p) / len(passed) if passed else 0,
            "total_evaluated": len(results["details"]),
            "tool_error_rate": results["tool_errors"] / results["total"] if results["total"] else 0,
            "agent_error_rate": results["agent_errors"] / results["total"] if results["total"] else 0,
            "p50_latency_ms": int(sorted(latencies)[len(latencies) // 2]) if latencies else 0,
            "p95_latency_ms": int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0,
        }

        # Per-category aggregation (mcp_agents_epam style).
        per_cat = defaultdict(lambda: {"composite": [], "relevancy": [], "groundedness": [],
                                         "keyword": [], "tool_correct": 0, "passed": 0, "n": 0})
        for d in results["details"]:
            cat = d["category"]
            per_cat[cat]["composite"].append(d["composite_score"])
            per_cat[cat]["relevancy"].append(d["relevancy_score"])
            per_cat[cat]["groundedness"].append(d["groundedness_score"])
            per_cat[cat]["keyword"].append(d["keyword_coverage"])
            per_cat[cat]["passed"] += int(d["passed"])
            per_cat[cat]["n"] += 1
        # Aggregate to means
        results["per_category"] = {
            cat: {
                "n": v["n"],
                "avg_composite": round(sum(v["composite"]) / v["n"], 3) if v["n"] else 0,
                "avg_relevancy": round(sum(v["relevancy"]) / v["n"], 2) if v["n"] else 0,
                "avg_groundedness": round(sum(v["groundedness"]) / v["n"], 2) if v["n"] else 0,
                "avg_keyword_coverage": round(sum(v["keyword"]) / v["n"], 2) if v["n"] else 0,
                "pass_rate": round(v["passed"] / v["n"], 2) if v["n"] else 0,
            }
            for cat, v in per_cat.items()
        }

        return results

    def generate_report(self, results: dict) -> str:
        if "error" in results:
            return f"Evaluation Error: {results['error']}"

        summary = results.get("summary", {})
        lines = [
            "=" * 60,
            "DISASTER CHATBOT EVALUATION REPORT",
            "=" * 60,
            f"Total questions evaluated: {summary.get('total_evaluated', 0)}",
            "",
            "OVERALL METRICS:",
            f"  Composite Score (avg):        {summary.get('avg_composite_score', 0):.3f} / 1.00",
            f"  Pass Rate:                    {summary.get('pass_rate', 0):.1%}",
            f"  Answer Relevancy (avg):       {summary.get('avg_answer_relevancy', 0):.2f} / 5.00",
            f"  Factual Groundedness (avg):   {summary.get('avg_factual_groundedness', 0):.2f} / 5.00",
            f"  Keyword Coverage (avg):       {summary.get('avg_keyword_coverage', 0):.2%}",
            (f"  Context Precision (avg):      {summary['avg_context_precision']:.2%}  "
             f"(over {summary['context_precision_n']} RAG-routed questions)"
             if summary.get('avg_context_precision') is not None
             else "  Context Precision (avg):      n/a (no RAG-routed questions in this run)"),
            f"  Tool Selection Accuracy:      {summary.get('tool_selection_accuracy', 0):.1%}",
            f"  Tool Error Rate:              {summary.get('tool_error_rate', 0):.1%}",
            f"  Agent Error Rate:             {summary.get('agent_error_rate', 0):.1%}",
            f"  Latency p50/p95 (ms):         {summary.get('p50_latency_ms', 0)} / {summary.get('p95_latency_ms', 0)}",
            "",
            "PER-CATEGORY BREAKDOWN:",
        ]
        for cat, m in results.get("per_category", {}).items():
            lines.append(f"  [{cat}] n={m['n']}  composite={m['avg_composite']:.3f}  pass={m['pass_rate']:.0%}  relevancy={m['avg_relevancy']:.2f}  groundedness={m['avg_groundedness']:.2f}  keyword={m['avg_keyword_coverage']:.2f}")
        lines += [
            "",
            "DETAIL BY QUESTION:",
        ]

        for i, detail in enumerate(results.get("details", []), 1):
            lines.append(f"\n  Q{i}: {detail['question']}")
            lines.append(f"      Category: {detail.get('category', '-')}")
            lines.append(f"      Tools used: {detail.get('tools_used', [])}")
            lines.append(
                f"      Relevancy: {detail.get('relevancy_score', 0)}/5  "
                f"Groundedness: {detail.get('groundedness_score', 0)}/5  "
                f"Keyword: {detail.get('keyword_coverage', 0):.0%}"
            )
            lines.append(
                f"      Composite: {detail.get('composite_score', 0):.3f} "
                f"(>= {detail.get('min_required', 0):.2f}) -> {'PASS' if detail.get('passed') else 'FAIL'}"
            )
            if detail.get('relevancy_reasoning'):
                lines.append(f"      Reasoning: {detail['relevancy_reasoning']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
