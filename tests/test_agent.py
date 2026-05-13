from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agent as agent_module


class TestDisasterAgent:
    def _create_agent(self):
        with patch.object(agent_module, "ChatOllama"), \
             patch.object(agent_module, "DisasterCSVTool"), \
             patch.object(agent_module, "NASAEONETTool"), \
             patch.object(agent_module, "GDACSTool"), \
             patch.object(agent_module, "DisasterRAGEngine"), \
             patch.object(agent_module, "create_react_agent", return_value=MagicMock()):
            return agent_module.DisasterAgent()

    def test_init_creates_tools(self):
        agent = self._create_agent()
        assert len(agent.tools) == 5  # csv, nasa, gdacs, rag, classify
        tool_names = [t.name for t in agent.tools]
        assert "query_disaster_csv" in tool_names
        assert "search_nasa_events" in tool_names
        assert "search_gdacs_events" in tool_names
        assert "query_disaster_knowledge" in tool_names
        assert "classify_disaster_image" in tool_names

    def test_chat_returns_dict(self):
        agent = self._create_agent()
        agent.graph = MagicMock()
        agent.graph.invoke.return_value = {
            "messages": [AIMessage(content="Test answer")],
        }
        result = agent.chat("test question")
        assert "answer" in result
        assert result["answer"] == "Test answer"
        assert "tools_used" in result

    def test_chat_handles_error(self):
        agent = self._create_agent()
        agent.graph = MagicMock()
        agent.graph.invoke.side_effect = Exception("LLM error")
        result = agent.chat("test question")
        assert "Error" in result["answer"]

    def test_chat_extracts_tools_used(self):
        agent = self._create_agent()
        agent.graph = MagicMock()
        tool_msg = AIMessage(content="", tool_calls=[{"name": "query_disaster_csv", "args": {}, "id": "1"}])
        final_msg = AIMessage(content="Final answer")
        agent.graph.invoke.return_value = {
            "messages": [tool_msg, final_msg],
        }
        result = agent.chat("How many disasters?")
        assert "query_disaster_csv" in result["tools_used"]

    def test_reset_memory(self):
        agent = self._create_agent()
        agent.chat_history = [MagicMock()]
        agent.reset_memory()
        assert agent.chat_history == []

    def test_index_knowledge_base(self):
        agent = self._create_agent()
        agent.rag_engine.index_disaster_data.return_value = "Indexed 100 documents"
        result = agent.index_knowledge_base()
        assert "Indexed" in result
