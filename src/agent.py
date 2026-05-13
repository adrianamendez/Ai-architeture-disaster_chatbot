import logging
import re
import time

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS, LLM_OUTPUT_LIMITS
from src.csv_tool import DisasterCSVTool
from src.nasa_eonet_tool import NASAEONETTool
from src.gdacs_tool import GDACSTool
from src.rag_engine import DisasterRAGEngine

logger = logging.getLogger(__name__)


def _safe_tool_call(name: str, fn, *args, **kwargs) -> str:
    """Wrap a tool call so a failure returns a graceful message instead of crashing the agent."""
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        logger.info("tool=%s status=ok latency_ms=%d", name, int((time.time() - t0) * 1000))
        return out if isinstance(out, str) else str(out)
    except Exception as e:
        logger.exception("tool=%s status=error err=%s", name, e)
        return f"[tool '{name}' failed: {e}. The assistant should try a different approach.]"


# OWASP LLM01 — input-side prompt-injection detector. Lightweight: catches obvious
# jailbreak attempts before they reach the agent. Not exhaustive, but real and enforced.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(prior|previous|all)\s+instructions", re.I),
    re.compile(r"system\s*prompt|initialization\s+preamble", re.I),
    re.compile(r"reveal.*(key|token|secret|password)|admin\s*token", re.I),
    re.compile(r"act\s+as\s+(?:a\s+)?(?:dan|developer|root|admin)", re.I),
    re.compile(r"</?(?:system|admin|root)>", re.I),
]
MAX_INPUT_CHARS = 4000


def _is_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


SYSTEM_PROMPT = (
    "You are a Natural Disaster Intelligence Assistant. You help users understand "
    "natural disasters using historical data, real-time event feeds, and image analysis.\n\n"
    "Guidelines:\n"
    "- Use query_disaster_csv for historical statistics, counts, comparisons, trends\n"
    "- Use search_nasa_events for current geospatial event tracking from NASA EONET\n"
    "- Use search_gdacs_events when humanitarian-impact level (green/orange/red) matters; "
    "GDACS complements NASA by adding humanitarian alert scoring\n"
    "- Prefer GDACS for floods, cyclones and earthquakes when the user asks about severity, "
    "alert levels or humanitarian impact; prefer NASA EONET for raw geospatial event tracking\n"
    "- Use query_disaster_knowledge for general knowledge questions about disasters\n"
    "- Use classify_disaster_image when given an image path to analyze\n"
    "- Combine multiple tools when needed for comprehensive answers\n"
    "- Always provide specific numbers and data when available"
)


class DisasterAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=MODELS["chat"]["name"],
            temperature=MODELS["chat"]["temperature"],
            num_predict=LLM_OUTPUT_LIMITS["chat_num_predict"],
        )
        self.csv_tool_instance = DisasterCSVTool()
        self.nasa_tool_instance = NASAEONETTool()
        self.gdacs_tool_instance = GDACSTool()
        self.rag_engine = DisasterRAGEngine()
        self.tools = self._create_tools()
        self.graph = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )
        self.chat_history = []

    def _create_tools(self) -> list:
        csv_inst = self.csv_tool_instance
        nasa_inst = self.nasa_tool_instance
        gdacs_inst = self.gdacs_tool_instance
        rag_inst = self.rag_engine

        @tool
        def query_disaster_csv(question: str) -> str:
            """Query historical disaster databases with natural language.
            Use for statistics, casualties, economic losses, country comparisons,
            and trends from 1900-2024. Input: a question about disaster data."""
            return _safe_tool_call("query_disaster_csv", csv_inst.query, question)

        @tool
        def search_nasa_events(query: str) -> str:
            """Search current and recent natural disaster events from NASA EONET.
            Use for real-time or recent events like active wildfires, earthquakes, floods.
            Input: a natural language search query."""
            return _safe_tool_call("search_nasa_events", nasa_inst.search_events, query)

        @tool
        def search_gdacs_events(query: str) -> str:
            """Search current natural disaster events from GDACS (Global Disaster
            Alert and Coordination System). Adds humanitarian-impact alert levels
            (green/orange/red) on top of NASA EONET. Best for queries that mention
            severity, alert level, or humanitarian impact (floods, cyclones, earthquakes).
            Input: a natural language search query."""
            return _safe_tool_call("search_gdacs_events", gdacs_inst.search_events, query)

        @tool
        def query_disaster_knowledge(question: str) -> str:
            """Query disaster knowledge base using semantic search (RAG).
            Use for general questions about disasters, response patterns, impacts.
            Input: a question about natural disasters."""
            return _safe_tool_call("query_disaster_knowledge", rag_inst.query, question)

        @tool
        def classify_disaster_image(image_path: str) -> str:
            """Classify an image to detect if it shows a natural disaster.
            Returns disaster type (fire, flood, landslide, earthquake, smoke, normal)
            with confidence. Input: absolute file path to the image."""
            return _safe_tool_call("classify_disaster_image", rag_inst.classify_image, image_path)

        return [query_disaster_csv, search_nasa_events, search_gdacs_events,
                query_disaster_knowledge, classify_disaster_image]

    def chat(self, message: str) -> dict:
        t0 = time.time()
        logger.info("agent.chat start msg_len=%d", len(message))

        # OWASP LLM01 + LLM10 — enforce input limits and reject obvious injections.
        if not isinstance(message, str) or not message.strip():
            return {"answer": "Empty input — please provide a question.",
                    "tools_used": [], "intermediate_steps": [],
                    "blocked_reason": "empty_input", "latency_ms": 0}
        if len(message) > MAX_INPUT_CHARS:
            logger.warning("agent.chat blocked: input over %d chars", MAX_INPUT_CHARS)
            return {"answer": f"Input exceeds the maximum of {MAX_INPUT_CHARS} characters.",
                    "tools_used": [], "intermediate_steps": [],
                    "blocked_reason": "input_too_long", "latency_ms": 0}
        if _is_prompt_injection(message):
            logger.warning("agent.chat blocked: prompt-injection pattern matched")
            return {"answer": "Your request was blocked because it matches known prompt-injection patterns. "
                              "Rephrase your disaster-related question without meta-instructions.",
                    "tools_used": [], "intermediate_steps": [],
                    "blocked_reason": "prompt_injection", "latency_ms": 0}

        try:
            self.chat_history.append(HumanMessage(content=message))
            result = self.graph.invoke({"messages": self.chat_history})
            messages = result.get("messages", [])

            answer = ""
            tools_used = []
            for msg in messages:
                if hasattr(msg, "content") and isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc["name"])
                    elif msg.content:
                        answer = msg.content

            if not answer and messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    answer = last.content

            self.chat_history.append(AIMessage(content=answer))
            logger.info(
                "agent.chat done latency_ms=%d tools=%s answer_len=%d",
                int((time.time() - t0) * 1000), tools_used, len(answer),
            )

            return {
                "answer": answer,
                "tools_used": tools_used,
                "intermediate_steps": [
                    {"tool": m.name, "content": m.content}
                    for m in messages
                    if hasattr(m, "name") and m.name
                ],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            logger.exception("agent.chat fatal err=%s", e)
            return {
                "answer": f"Error processing your question: {e}",
                "intermediate_steps": [],
                "tools_used": [],
                "latency_ms": int((time.time() - t0) * 1000),
            }

    def index_knowledge_base(self, force: bool = False) -> str:
        return self.rag_engine.index_disaster_data(force=force)

    def reset_memory(self):
        self.chat_history = []
