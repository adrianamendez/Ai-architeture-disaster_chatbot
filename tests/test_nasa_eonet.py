import pytest
from unittest.mock import patch, MagicMock
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.nasa_eonet_tool import NASAEONETTool


MOCK_EVENTS_RESPONSE = {
    "events": [
        {
            "id": "EONET_001",
            "title": "Wildfire in California",
            "categories": [{"id": "wildfires", "title": "Wildfires"}],
            "sources": [{"url": "https://example.com"}],
            "geometry": [
                {
                    "date": "2024-07-15T00:00:00Z",
                    "coordinates": [-120.5, 37.8],
                }
            ],
        },
        {
            "id": "EONET_002",
            "title": "Earthquake in Japan",
            "categories": [{"id": "earthquakes", "title": "Earthquakes"}],
            "sources": [],
            "geometry": [
                {
                    "date": "2024-07-14T12:00:00Z",
                    "coordinates": [139.7, 35.6],
                }
            ],
        },
    ]
}

MOCK_CATEGORIES_RESPONSE = {
    "categories": [
        {"id": "wildfires", "title": "Wildfires"},
        {"id": "earthquakes", "title": "Earthquakes"},
        {"id": "floods", "title": "Floods"},
    ]
}


@pytest.fixture
def nasa_tool():
    return NASAEONETTool()


class TestNASAEONETTool:
    def test_init(self, nasa_tool):
        assert nasa_tool.base_url == "https://eonet.gsfc.nasa.gov/api/v3"
        assert "wildfires" in nasa_tool.categories

    @patch("src.nasa_eonet_tool.httpx.Client")
    def test_get_events_success(self, mock_client_cls, nasa_tool):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EVENTS_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = nasa_tool.get_events()
        assert "Wildfire in California" in result
        assert "Earthquake in Japan" in result
        assert "EONET_001" in result

    @patch("src.nasa_eonet_tool.httpx.Client")
    def test_get_events_with_category(self, mock_client_cls, nasa_tool):
        mock_response = MagicMock()
        mock_response.json.return_value = {"events": [MOCK_EVENTS_RESPONSE["events"][0]]}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = nasa_tool.get_events(category="wildfires")
        assert "Wildfire" in result

    @patch("src.nasa_eonet_tool.httpx.Client")
    def test_get_events_empty(self, mock_client_cls, nasa_tool):
        mock_response = MagicMock()
        mock_response.json.return_value = {"events": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = nasa_tool.get_events()
        assert "No active" in result

    @patch("src.nasa_eonet_tool.httpx.Client")
    def test_get_events_api_error(self, mock_client_cls, nasa_tool):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = nasa_tool.get_events()
        assert "Connection error" in result

    @patch("src.nasa_eonet_tool.httpx.Client")
    def test_get_categories(self, mock_client_cls, nasa_tool):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_CATEGORIES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = nasa_tool.get_categories()
        assert "Wildfires" in result
        assert "Earthquakes" in result

    def test_search_events_detects_category(self, nasa_tool):
        with patch.object(nasa_tool, "get_events", return_value="mocked") as mock_get:
            nasa_tool.search_events("active wildfires in 30 days")
            mock_get.assert_called_once_with(
                category="wildfires", status="open", limit=10, days=30
            )

    def test_search_events_detects_closed(self, nasa_tool):
        with patch.object(nasa_tool, "get_events", return_value="mocked") as mock_get:
            nasa_tool.search_events("past earthquakes")
            mock_get.assert_called_once_with(
                category="earthquakes", status="closed", limit=10, days=None
            )

    def test_search_events_default_open(self, nasa_tool):
        with patch.object(nasa_tool, "get_events", return_value="mocked") as mock_get:
            nasa_tool.search_events("any events")
            mock_get.assert_called_once_with(
                category=None, status="open", limit=10, days=None
            )
