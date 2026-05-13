from fastmcp import FastMCP
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.csv_tool import DisasterCSVTool
from src.nasa_eonet_tool import NASAEONETTool
from src.gdacs_tool import GDACSTool
from src.rag_engine import DisasterRAGEngine

mcp = FastMCP("disaster-intelligence")

csv_tool = None
nasa_tool = None
gdacs_tool = None
rag_engine = None


def _get_csv_tool():
    global csv_tool
    if csv_tool is None:
        csv_tool = DisasterCSVTool()
    return csv_tool


def _get_nasa_tool():
    global nasa_tool
    if nasa_tool is None:
        nasa_tool = NASAEONETTool()
    return nasa_tool


def _get_gdacs_tool():
    global gdacs_tool
    if gdacs_tool is None:
        gdacs_tool = GDACSTool()
    return gdacs_tool


def _get_rag_engine():
    global rag_engine
    if rag_engine is None:
        rag_engine = DisasterRAGEngine()
    return rag_engine


@mcp.tool()
def query_disaster_csv(question: str) -> str:
    """Query historical disaster databases using natural language.
    Supports questions about disaster statistics, casualties, economic losses,
    country comparisons, and temporal trends from 1900-2024.
    """
    tool = _get_csv_tool()
    return tool.query(question)


@mcp.tool()
def get_disaster_statistics(dataset: str = "global_response") -> str:
    """Get summary statistics for a disaster dataset.
    Available datasets: global_response, emdat_1970, emdat_1900
    """
    tool = _get_csv_tool()
    return tool.get_disaster_stats(dataset)


@mcp.tool()
def get_nasa_events(
    category: Optional[str] = None,
    status: str = "open",
    limit: int = 10,
    days: Optional[int] = None,
) -> str:
    """Get current natural disaster events from NASA EONET.
    Categories: wildfires, severe_storms, volcanoes, earthquakes, floods, landslides, drought.
    Status: open (active), closed (past), all.
    """
    tool = _get_nasa_tool()
    return tool.get_events(category=category, status=status, limit=limit, days=days)


@mcp.tool()
def search_nasa_events(query: str, limit: int = 10) -> str:
    """Search NASA EONET events using a natural language query.
    Examples: 'active wildfires', 'recent earthquakes', 'floods in the last 30 days'
    """
    tool = _get_nasa_tool()
    return tool.search_events(query, limit=limit)


@mcp.tool()
def get_gdacs_events(
    category: str = "all",
    window: str = "24h",
    limit: int = 10,
    min_alert: Optional[str] = None,
) -> str:
    """Get current natural disaster events from GDACS (Global Disaster Alert
    and Coordination System). Categories: all, earthquakes, cyclones, floods.
    Windows: 24h, 48h, 7d, 3m. min_alert: green/orange/red — filter by humanitarian-impact level.
    GDACS complements NASA EONET by adding alert-level humanitarian scoring.
    """
    tool = _get_gdacs_tool()
    return tool.get_events(category=category, window=window, limit=limit, min_alert=min_alert)


@mcp.tool()
def search_gdacs_events(query: str, limit: int = 10) -> str:
    """Search GDACS events using a natural-language query.
    Examples: 'recent earthquakes', 'red alert floods', 'cyclones in the last week'.
    """
    tool = _get_gdacs_tool()
    return tool.search_events(query, limit=limit)


@mcp.tool()
def classify_disaster_image(image_path: str) -> str:
    """Classify an image to determine if it shows a natural disaster
    and identify the type (fire, flood, landslide, earthquake, smoke, or normal).
    """
    engine = _get_rag_engine()
    result = engine.classify_image(image_path)
    return str(result)


@mcp.tool()
def query_disaster_knowledge(question: str) -> str:
    """Query the disaster knowledge base using RAG (Retrieval Augmented Generation).
    Uses semantic search over indexed disaster data to find relevant information
    and generate comprehensive answers.
    """
    engine = _get_rag_engine()
    return engine.query(question)


@mcp.tool()
def index_disaster_data(force: bool = False) -> str:
    """Index disaster CSV data into the vector database for RAG queries.
    Run this once before using query_disaster_knowledge.
    Set force=True to re-index from scratch.
    """
    engine = _get_rag_engine()
    return engine.index_disaster_data(force=force)


if __name__ == "__main__":
    mcp.run(transport="stdio")
