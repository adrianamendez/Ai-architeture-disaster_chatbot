import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.csv_tool import DisasterCSVTool


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": ["2021-01-01", "2021-06-15", "2022-03-10"],
        "country": ["Brazil", "Japan", "Brazil"],
        "disaster_type": ["Earthquake", "Flood", "Earthquake"],
        "severity_index": [5.5, 3.2, 7.1],
        "casualties": [100, 50, 200],
        "economic_loss_usd": [1000000, 500000, 2000000],
    })


@pytest.fixture
def csv_tool(sample_df):
    with patch.object(DisasterCSVTool, "_load_csvs"):
        tool = DisasterCSVTool()
        tool.dataframes = {"global_response": sample_df}
    return tool


class TestDisasterCSVTool:
    def test_init_loads_dataframes(self, csv_tool):
        assert "global_response" in csv_tool.dataframes
        assert len(csv_tool.dataframes["global_response"]) == 3

    def test_get_schema_summary(self, csv_tool):
        schema = csv_tool.get_schema_summary()
        assert "global_response" in schema
        assert "3 rows" in schema
        assert "country" in schema

    def test_get_available_datasets(self, csv_tool):
        info = csv_tool.get_available_datasets()
        assert "global_response" in info
        assert "3 rows" in info

    def test_get_disaster_stats(self, csv_tool):
        stats = csv_tool.get_disaster_stats("global_response")
        assert "Disaster types: 2" in stats
        assert "Earthquake" in stats

    def test_get_disaster_stats_missing_dataset(self, csv_tool):
        result = csv_tool.get_disaster_stats("nonexistent")
        assert "not found" in result

    def test_extract_code_from_code_block(self, csv_tool):
        llm_response = "```python\nresult = df.head()\n```"
        code = csv_tool._extract_code(llm_response)
        assert "result = df.head()" in code

    def test_extract_code_from_plain_text(self, csv_tool):
        llm_response = "result = df.head()"
        code = csv_tool._extract_code(llm_response)
        assert "result = df.head()" in code

    def test_safe_execute_basic(self, csv_tool):
        code = "result = len(self.dataframes['global_response'])"
        result = csv_tool._safe_execute(code)
        assert "3" in result

    def test_safe_execute_blocks_imports(self, csv_tool):
        code = "import os\nresult = os.listdir('.')"
        result = csv_tool._safe_execute(code)
        assert "unsafe" in result.lower() or "error" in result.lower()

    def test_safe_execute_blocks_exec(self, csv_tool):
        code = "exec('print(1)')"
        result = csv_tool._safe_execute(code)
        assert "unsafe" in result.lower() or "error" in result.lower()

    def test_safe_execute_handles_error(self, csv_tool):
        code = "result = 1 / 0"
        result = csv_tool._safe_execute(code)
        assert "Error" in result

    @patch("src.csv_tool.requests.post")
    def test_query_calls_ollama(self, mock_post, csv_tool):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "result = len(self.dataframes['global_response'])"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        result = csv_tool.query("How many records?")
        assert "3" in result
        mock_post.assert_called_once()

    @patch("src.csv_tool.requests.post")
    def test_query_handles_api_error(self, mock_post, csv_tool):
        mock_post.side_effect = Exception("Connection refused")
        result = csv_tool.query("How many records?")
        assert "Error" in result
