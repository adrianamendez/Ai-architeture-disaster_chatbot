from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMCPServerTools:
    @patch("mcp_server.server._get_csv_tool")
    def test_query_disaster_csv(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.query.return_value = "Result: 42 disasters"
        mock_get_tool.return_value = mock_tool

        from mcp_server.server import query_disaster_csv
        result = query_disaster_csv("How many disasters?")
        assert "42" in result
        mock_tool.query.assert_called_once_with("How many disasters?")

    @patch("mcp_server.server._get_csv_tool")
    def test_get_disaster_statistics(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.get_disaster_stats.return_value = "Disaster types: 5"
        mock_get_tool.return_value = mock_tool

        from mcp_server.server import get_disaster_statistics
        result = get_disaster_statistics("global_response")
        assert "Disaster types" in result

    @patch("mcp_server.server._get_nasa_tool")
    def test_get_nasa_events(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.get_events.return_value = "Wildfire in California"
        mock_get_tool.return_value = mock_tool

        from mcp_server.server import get_nasa_events
        result = get_nasa_events(category="wildfires")
        assert "Wildfire" in result

    @patch("mcp_server.server._get_nasa_tool")
    def test_search_nasa_events(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.search_events.return_value = "Active events found"
        mock_get_tool.return_value = mock_tool

        from mcp_server.server import search_nasa_events
        result = search_nasa_events("active wildfires")
        assert "Active" in result

    @patch("mcp_server.server._get_rag_engine")
    def test_classify_disaster_image(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.classify_image.return_value = {
            "disaster_type": "fire",
            "confidence": "high",
        }
        mock_get_engine.return_value = mock_engine

        from mcp_server.server import classify_disaster_image
        result = classify_disaster_image("/path/to/image.jpg")
        assert "fire" in result

    @patch("mcp_server.server._get_rag_engine")
    def test_query_disaster_knowledge(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.query.return_value = "Earthquakes are tectonic events"
        mock_get_engine.return_value = mock_engine

        from mcp_server.server import query_disaster_knowledge
        result = query_disaster_knowledge("What are earthquakes?")
        assert "tectonic" in result

    @patch("mcp_server.server._get_rag_engine")
    def test_index_disaster_data(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_engine.index_disaster_data.return_value = "Indexed 500 documents"
        mock_get_engine.return_value = mock_engine

        from mcp_server.server import index_disaster_data
        result = index_disaster_data(force=False)
        assert "Indexed" in result


class TestMCPServerLazyInit:
    def test_csv_tool_lazy_init(self):
        import mcp_server.server as srv
        srv.csv_tool = None
        with patch("mcp_server.server.DisasterCSVTool") as mock_cls:
            mock_cls.return_value = MagicMock()
            tool = srv._get_csv_tool()
            assert tool is not None
            mock_cls.assert_called_once()

    def test_nasa_tool_lazy_init(self):
        import mcp_server.server as srv
        srv.nasa_tool = None
        with patch("mcp_server.server.NASAEONETTool") as mock_cls:
            mock_cls.return_value = MagicMock()
            tool = srv._get_nasa_tool()
            assert tool is not None
            mock_cls.assert_called_once()

    def test_rag_engine_lazy_init(self):
        import mcp_server.server as srv
        srv.rag_engine = None
        with patch("mcp_server.server.DisasterRAGEngine") as mock_cls:
            mock_cls.return_value = MagicMock()
            engine = srv._get_rag_engine()
            assert engine is not None
            mock_cls.assert_called_once()
