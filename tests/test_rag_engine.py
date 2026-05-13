import pytest
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_engine import DisasterRAGEngine


@pytest.fixture
def rag_engine():
    with patch("src.rag_engine.SentenceTransformer") as mock_st, \
         patch("src.rag_engine.chromadb.PersistentClient") as mock_chroma:
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])
        mock_st.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        engine = DisasterRAGEngine()
        engine.collection = mock_collection
        return engine


class TestDisasterRAGEngine:
    def test_init(self, rag_engine):
        assert rag_engine.embedding_model is not None
        assert rag_engine.collection is not None
        assert rag_engine._indexed is False

    def test_row_to_document_global_response(self, rag_engine):
        import pandas as pd
        row = pd.Series({
            "date": "2021-01-01",
            "country": "Brazil",
            "disaster_type": "Earthquake",
            "severity_index": 5.5,
            "casualties": 100,
            "economic_loss_usd": 1000000,
            "response_time_hours": 10,
        })
        doc = rag_engine._row_to_document(row, "global_response")
        assert "Brazil" in doc
        assert "Earthquake" in doc
        assert "100" in doc

    def test_row_to_document_empty(self, rag_engine):
        import pandas as pd
        row = pd.Series(dtype=float)
        doc = rag_engine._row_to_document(row, "global_response")
        assert doc is None

    def test_query_not_indexed(self, rag_engine):
        result = rag_engine.query("test question")
        assert "not indexed" in result.lower()

    @patch("src.rag_engine.requests.post")
    def test_query_indexed(self, mock_post, rag_engine):
        rag_engine._indexed = True
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.1] * 384
        rag_engine.embedding_model.encode.return_value = mock_array

        rag_engine.collection.query.return_value = {
            "documents": [["doc1 about floods", "doc2 about earthquakes"]],
            "metadatas": [[{"source": "test"}]],
        }

        # Mock the cross-encoder reranker so it returns scores above min_score for both docs
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.7]
        rag_engine._reranker = mock_reranker

        mock_post.return_value = MagicMock(
            json=lambda: {"response": "Floods are common disasters."},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = rag_engine.query("What are common disasters?")
        assert "Floods" in result or "common" in result.lower()

    def test_parse_classification_json(self, rag_engine):
        raw = '{"disaster_type": "fire", "confidence": "high", "description": "forest fire"}'
        result = rag_engine._parse_classification(raw)
        assert result["disaster_type"] == "fire"
        assert result["confidence"] == "high"

    def test_parse_classification_text_fallback(self, rag_engine):
        raw = "This image shows a severe flood in a residential area"
        result = rag_engine._parse_classification(raw)
        assert result["disaster_type"] == "flood"

    def test_parse_classification_unknown(self, rag_engine):
        raw = "I cannot determine what this image shows"
        result = rag_engine._parse_classification(raw)
        assert result["disaster_type"] == "unknown"

    def test_parse_classification_json_in_text(self, rag_engine):
        raw = 'Some text before {"disaster_type": "earthquake", "confidence": "medium", "description": "damage"} and after'
        result = rag_engine._parse_classification(raw)
        assert result["disaster_type"] == "earthquake"

    @patch("src.rag_engine.requests.post")
    @patch("src.rag_engine.Image.open")
    def test_classify_image(self, mock_open, mock_post, rag_engine, tmp_path):
        test_img = tmp_path / "test.jpg"
        test_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.mode = "RGB"
        mock_img.save = MagicMock()
        mock_open.return_value = mock_img

        mock_post.return_value = MagicMock(
            json=lambda: {
                "response": '{"disaster_type": "fire", "confidence": "high", "description": "wildfire"}'
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = rag_engine.classify_image(str(test_img))
        assert result["disaster_type"] == "fire"

    def test_classify_image_not_found(self, rag_engine):
        result = rag_engine.classify_image("/nonexistent/image.jpg")
        assert "error" in result

    def test_get_sample_images(self, rag_engine):
        with patch("src.rag_engine.DISASTER_IMAGES_DIR") as mock_dir:
            mock_fire_dir = MagicMock()
            mock_fire_dir.exists.return_value = True
            mock_fire_dir.glob.return_value = [
                Path("/fake/fire1.jpg"),
                Path("/fake/fire2.jpg"),
                Path("/fake/fire3.jpg"),
            ]
            mock_dir.__truediv__ = MagicMock(return_value=mock_fire_dir)
            result = rag_engine.get_sample_images("fire", n=2)
            assert len(result) <= 2
