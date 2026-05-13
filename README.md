# Natural Disaster Intelligence Chatbot

Final project combining **RAG**, **MCP server**, and **agents** into a multimodal
chatbot for natural disaster intelligence. Inherits work from three previous
projects (`rag-document-analyzer`, `mcp_agents_epam`, `ia-security-task`).

## Two ways to run this

### 1. Streamlit dashboard (lightweight, no Ollama required)

The dashboard renders pre-computed results from `streamlit_data.py`. It does NOT
call Ollama, NASA, GDACS, or load any model — perfect for a quick demo or for
deploying to Streamlit Community Cloud.

```bash
pip install -r requirements.txt   # slim: streamlit, plotly, pandas, numpy
streamlit run streamlit_app.py
```

### 2. Full notebook (requires Ollama running locally)

To reproduce every cell in `disaster_chatbot.ipynb` you need the full stack:
local Ollama with the `llama3.2` and `llava` models, ChromaDB, embeddings,
the agent, the evaluator, etc.

```bash
pip install -r requirements-notebook.txt
ollama pull llama3.2 && ollama pull llava
# Download the Kaggle datasets per info_rag/README.md
jupyter notebook disaster_chatbot.ipynb
```

## Project layout

```
.
├── disaster_chatbot.ipynb     Main deliverable (10 sections)
├── streamlit_app.py           Dashboard (no live LLM calls)
├── streamlit_data.py          Hardcoded fixtures for the dashboard
├── config.py                  Shared configuration (paths, models, thresholds)
├── requirements.txt           Slim deps for the Streamlit deploy
├── requirements-notebook.txt  Full deps for running the notebook
├── src/                       Agent, RAG engine, CSV/NASA/GDACS tools, evaluator
├── mcp_server/                FastMCP stdio server exposing 8 tools
├── tests/                     75 unit tests (pytest)
├── eval_data/                 12-question benchmark
├── info_rag/                  Datasets (gitignored, see info_rag/README.md)
└── test_images_colombia/      Real-world out-of-distribution test images
```

## Key features

- ReAct agent (LangGraph) routing across 5 read-only tools
- RAG with ChromaDB + Sentence-Transformers + cross-encoder reranking
- Multimodal classification with LLaVA via Ollama
- Two live disaster feeds: NASA EONET + GDACS (humanitarian alert levels)
- MCP server exposing all 8 tools over stdio
- Composite evaluation framework (relevancy, groundedness, keyword coverage,
  context precision, tool selection) inspired by RAGAS and `mcp_agents_epam`
- OWASP for LLMs 2025 controls enforced in code (prompt-injection guard,
  read-only tools, cited RAG sources, input/output caps)

## Tests

```bash
pip install -r requirements-notebook.txt
pytest tests/ -v
```

All 75 tests should pass (no Ollama required for the test suite — everything is
mocked).
