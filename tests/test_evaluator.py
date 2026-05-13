import pytest
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import DisasterChatbotEvaluator, context_precision, keyword_coverage


class TestRagasInspiredMetrics:
    def test_context_precision_full_match(self):
        chunks = [{"text": "flooding in united states"}, {"text": "us flood event"}]
        assert context_precision(chunks, ["united states", "flood"]) == 1.0

    def test_context_precision_partial(self):
        chunks = [
            {"text": "United States flooding event"},
            {"text": "Earthquake in Chile"},
            {"text": "Cyclone in Bangladesh"},
        ]
        assert context_precision(chunks, ["united states", "flood"]) == pytest.approx(1 / 3)

    def test_context_precision_empty_inputs(self):
        assert context_precision([], ["foo"]) == 1.0
        assert context_precision([{"text": "x"}], []) == 1.0

    def test_context_precision_accent_insensitive(self):
        chunks = [{"text": "huracan en mexico"}]
        assert context_precision(chunks, ["huracán"]) == 1.0

    def test_keyword_coverage_basic(self):
        assert keyword_coverage("This is about floods in Brazil", ["flood", "brazil"]) == 1.0
        assert keyword_coverage("No relevant words", ["flood", "brazil"]) == 0.0


@pytest.fixture
def eval_data(tmp_path):
    data = [
        {
            "question": "How many floods occurred in 2020?",
            "expected_tool": "query_disaster_csv",
            "reference_answer": "There were 150 floods in 2020.",
            "category": "csv_query",
        },
        {
            "question": "What are current active wildfires?",
            "expected_tool": "search_nasa_events",
            "reference_answer": "NASA tracks active wildfires globally.",
            "category": "nasa_events",
        },
    ]
    path = tmp_path / "eval_dataset.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def evaluator(eval_data):
    return DisasterChatbotEvaluator(eval_dataset_path=eval_data)


class TestDisasterChatbotEvaluator:
    def test_load_eval_data(self, evaluator):
        assert len(evaluator.eval_data) == 2
        assert evaluator.eval_data[0]["question"] == "How many floods occurred in 2020?"

    def test_load_missing_file(self):
        ev = DisasterChatbotEvaluator(eval_dataset_path="/nonexistent/file.json")
        assert ev.eval_data == []

    def test_evaluate_tool_selection_correct(self, evaluator):
        result = evaluator.evaluate_tool_selection(
            "test", ["query_disaster_csv"], "query_disaster_csv"
        )
        assert result["correct"] is True
        assert result["score"] == 5

    def test_evaluate_tool_selection_wrong(self, evaluator):
        result = evaluator.evaluate_tool_selection(
            "test", ["search_nasa_events"], "query_disaster_csv"
        )
        assert result["correct"] is False
        assert result["score"] == 1

    def test_evaluate_tool_selection_multiple_tools(self, evaluator):
        result = evaluator.evaluate_tool_selection(
            "test", ["search_nasa_events", "query_disaster_csv"], "query_disaster_csv"
        )
        assert result["correct"] is True

    @patch("src.evaluator.requests.post")
    def test_llm_judge_success(self, mock_post, evaluator):
        mock_post.return_value = MagicMock(
            json=lambda: {"response": '{"score": 4, "reasoning": "Good answer"}'},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = evaluator.evaluate_answer_relevancy("Question?", "Answer.")
        assert result["score"] == 4
        assert "Good" in result["reasoning"]

    @patch("src.evaluator.requests.post")
    def test_llm_judge_failure(self, mock_post, evaluator):
        mock_post.side_effect = Exception("Connection refused")
        result = evaluator.evaluate_answer_relevancy("Question?", "Answer.")
        assert result["score"] == 0

    @patch("src.evaluator.requests.post")
    def test_run_evaluation(self, mock_post, evaluator):
        mock_post.return_value = MagicMock(
            json=lambda: {"response": '{"score": 4, "reasoning": "Good"}'},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_agent = MagicMock()
        mock_agent.chat.return_value = {
            "answer": "Test answer",
            "intermediate_steps": [],
            "tools_used": ["query_disaster_csv"],
        }

        results = evaluator.run_evaluation(mock_agent)
        assert results["total"] == 2
        assert "summary" in results
        assert results["summary"]["total_evaluated"] == 2

    def test_run_evaluation_no_data(self):
        ev = DisasterChatbotEvaluator(eval_dataset_path="/nonexistent.json")
        results = ev.run_evaluation(MagicMock())
        assert "error" in results

    @patch("src.evaluator.requests.post")
    def test_generate_report(self, mock_post, evaluator):
        mock_post.return_value = MagicMock(
            json=lambda: {"response": '{"score": 4, "reasoning": "Good"}'},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_agent = MagicMock()
        mock_agent.chat.return_value = {
            "answer": "Answer",
            "intermediate_steps": [],
            "tools_used": ["query_disaster_csv"],
        }

        results = evaluator.run_evaluation(mock_agent)
        report = evaluator.generate_report(results)
        assert "EVALUATION REPORT" in report
        assert "Answer Relevancy" in report

    def test_generate_report_error(self, evaluator):
        report = evaluator.generate_report({"error": "No data"})
        assert "Error" in report
