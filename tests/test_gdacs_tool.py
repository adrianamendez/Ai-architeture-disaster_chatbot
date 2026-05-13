import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gdacs_tool import GDACSTool, GDACS_FEEDS


SAMPLE_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:gdacs="http://www.gdacs.org">
  <channel>
    <title>GDACS sample</title>
    <item>
      <title>Orange flood alert in Brazil</title>
      <link>https://www.gdacs.org/report.aspx?eventtype=FL&amp;eventid=42</link>
      <pubDate>Mon, 12 May 2026 12:00:00 GMT</pubDate>
      <gdacs:eventtype>FL</gdacs:eventtype>
      <gdacs:alertlevel>Orange</gdacs:alertlevel>
      <gdacs:country>Brazil</gdacs:country>
      <gdacs:severity>2</gdacs:severity>
    </item>
    <item>
      <title>Green earthquake (Magnitude 4.5M) in Chile</title>
      <link>https://www.gdacs.org/report.aspx?eventtype=EQ&amp;eventid=99</link>
      <pubDate>Mon, 12 May 2026 13:00:00 GMT</pubDate>
      <gdacs:eventtype>EQ</gdacs:eventtype>
      <gdacs:alertlevel>Green</gdacs:alertlevel>
      <gdacs:country>Chile</gdacs:country>
      <gdacs:severity>4.5</gdacs:severity>
    </item>
  </channel>
</rss>
"""


@pytest.fixture
def gdacs():
    return GDACSTool(base_url="https://www.gdacs.org/xml")


class TestGDACSTool:
    def test_get_available_feeds(self, gdacs):
        out = gdacs.get_available_feeds()
        assert "earthquakes" in out
        assert "floods" in out
        assert "alert levels" in out.lower()

    def test_get_events_invalid_feed(self, gdacs):
        out = gdacs.get_events(category="invalid", window="999h")
        assert "Invalid GDACS feed" in out

    @patch("src.gdacs_tool._fetch_rss")
    def test_get_events_parses_rss(self, mock_fetch, gdacs):
        mock_fetch.return_value = SAMPLE_RSS
        out = gdacs.get_events(category="all", window="24h", limit=10)
        assert "Orange flood alert in Brazil" in out
        assert "[ORANGE]" in out
        assert "[GREEN]" in out
        assert "Type: FL" in out
        assert "Type: EQ" in out

    @patch("src.gdacs_tool._fetch_rss")
    def test_get_events_filters_by_min_alert(self, mock_fetch, gdacs):
        mock_fetch.return_value = SAMPLE_RSS
        out = gdacs.get_events(category="all", window="24h", min_alert="orange")
        assert "Orange flood alert in Brazil" in out
        assert "Green earthquake" not in out

    @patch("src.gdacs_tool._fetch_rss")
    def test_search_events_routes_earthquakes(self, mock_fetch, gdacs):
        mock_fetch.return_value = SAMPLE_RSS
        gdacs.search_events("recent earthquakes today")
        args, _ = mock_fetch.call_args
        assert "rss_eq" in args[0]

    @patch("src.gdacs_tool._fetch_rss")
    def test_search_events_routes_floods(self, mock_fetch, gdacs):
        mock_fetch.return_value = SAMPLE_RSS
        gdacs.search_events("major floods this week")
        args, _ = mock_fetch.call_args
        assert "rss_fl" in args[0]

    @patch("src.gdacs_tool._fetch_rss")
    def test_search_events_extracts_alert_filter(self, mock_fetch, gdacs):
        mock_fetch.return_value = SAMPLE_RSS
        out = gdacs.search_events("red alert events")
        # red alert means only items with alertlevel=red survive; sample has none
        assert "No GDACS events" in out or "RED" in out

    @patch("src.gdacs_tool._fetch_rss")
    def test_get_events_handles_fetch_error(self, mock_fetch, gdacs):
        mock_fetch.side_effect = Exception("connection refused")
        out = gdacs.get_events(category="all", window="24h")
        assert "GDACS connection error" in out

    def test_feeds_catalog_completeness(self):
        # Sanity: every category/window combo we expose maps to a real filename string
        for (cat, win), fname in GDACS_FEEDS.items():
            assert fname.endswith(".xml")
            assert cat in {"all", "earthquakes", "cyclones", "floods"}
            assert win in {"24h", "48h", "7d", "3m"}
