import logging
import time
from typing import Optional
from pathlib import Path
import sys

import feedparser
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GDACS_BASE_URL, HTTP_RETRY

logger = logging.getLogger(__name__)


# Maps user-facing category + window into the actual feed filename hosted by GDACS.
# Source: https://gdacs.org/feed_reference.aspx
GDACS_FEEDS = {
    ("all", "24h"):           "rss_24h.xml",
    ("all", "7d"):            "rss_7d.xml",
    ("earthquakes", "24h"):   "rss_eq_24h.xml",
    ("earthquakes", "48h"):   "rss_eq_48h_med.xml",
    ("earthquakes", "3m"):    "rss_eq_5.5_3m.xml",
    ("cyclones", "7d"):       "rss_tc_7d.xml",
    ("cyclones", "3m"):       "rss_tc_3m.xml",
    ("floods", "7d"):         "rss_fl_7d.xml",
    ("floods", "3m"):         "rss_fl_3m.xml",
}

ALERT_LEVELS = {"green", "orange", "red"}


def _fetch_rss(url: str, timeout: int = 30) -> str:
    """GET an RSS feed with retry + backoff."""
    last_exc = None
    for attempt in range(1, HTTP_RETRY["attempts"] + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, headers={"Accept": "application/rss+xml, application/xml"})
                r.raise_for_status()
                logger.debug("GDACS GET %s ok (attempt=%d)", url, attempt)
                return r.text
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_exc = e
            sleep = HTTP_RETRY["backoff_seconds"] * (2 ** (attempt - 1))
            logger.warning("GDACS GET %s attempt=%d/%d failed: %s — backoff %.1fs",
                           url, attempt, HTTP_RETRY["attempts"], e, sleep)
            if attempt < HTTP_RETRY["attempts"]:
                time.sleep(sleep)
    raise last_exc


class GDACSTool:
    """Live disaster events from the Global Disaster Alert and Coordination System.

    GDACS publishes RSS feeds updated every ~6 minutes with humanitarian-impact
    classification (Green / Orange / Red alert levels). Complements NASA EONET,
    which focuses on geospatial event tracking without the humanitarian scoring.
    """

    def __init__(self, base_url: str = GDACS_BASE_URL):
        self.base_url = base_url

    def get_events(
        self,
        category: str = "all",
        window: str = "24h",
        limit: int = 10,
        min_alert: Optional[str] = None,
    ) -> str:
        """Return human-readable summary of GDACS events.

        category: one of all / earthquakes / cyclones / floods
        window:   one of 24h / 48h / 7d / 3m  (combinations limited by `GDACS_FEEDS`)
        limit:    max events to display
        min_alert: optional filter green / orange / red — drop events below this level
        """
        category = (category or "all").lower()
        window = (window or "24h").lower()
        feed_path = GDACS_FEEDS.get((category, window))
        if not feed_path:
            available = sorted({f"{c}/{w}" for (c, w) in GDACS_FEEDS})
            return (f"Invalid GDACS feed (category='{category}', window='{window}'). "
                    f"Available: {', '.join(available)}")

        url = f"{self.base_url}/{feed_path}"
        try:
            xml = _fetch_rss(url, timeout=HTTP_RETRY["request_timeout"])
        except Exception as e:
            logger.error("GDACS fetch failed: %s", e)
            return f"GDACS connection error: {e}"

        feed = feedparser.parse(xml)
        if feed.bozo and feed.entries == []:
            return f"GDACS returned a malformed feed: {feed.bozo_exception}"

        items = feed.entries
        if min_alert and min_alert.lower() in ALERT_LEVELS:
            order = {"green": 1, "orange": 2, "red": 3}
            threshold = order[min_alert.lower()]
            items = [
                e for e in items
                if order.get(self._extract_alert(e).lower(), 0) >= threshold
            ]

        items = items[:limit]
        if not items:
            return f"No GDACS events for category={category}, window={window}, min_alert={min_alert}."

        lines = [f"GDACS events ({category}, last {window}, n={len(items)})"]
        for e in items:
            title = e.get("title", "Unknown")
            published = e.get("published", "n/a")
            event_type = self._extract(e, "gdacs_eventtype") or "?"
            alert = self._extract_alert(e)
            severity = self._extract(e, "gdacs_severity") or "n/a"
            country = self._extract(e, "gdacs_country") or "n/a"
            link = e.get("link", "")
            lat = e.get("geo_lat") or e.get("where", {}).get("lat") if hasattr(e, "where") else None
            lon = e.get("geo_long")
            location = f"({lat}, {lon})" if lat and lon else "n/a"
            lines.append(
                f"- **{title}** [{alert.upper()}]\n"
                f"  Type: {event_type}  Country: {country}  Severity: {severity}\n"
                f"  Location: {location}  Published: {published}\n"
                f"  Link: {link}"
            )
        return "\n\n".join(lines)

    def search_events(self, query: str, limit: int = 10) -> str:
        """Best-effort routing from a natural-language query to a feed + filter."""
        q = (query or "").lower()

        if "earthquake" in q or "quake" in q or "seismic" in q:
            cat = "earthquakes"
            window = "24h" if "today" in q or "24" in q else "48h"
        elif "cyclone" in q or "hurricane" in q or "typhoon" in q or "storm" in q:
            cat = "cyclones"
            window = "7d"
        elif "flood" in q:
            cat = "floods"
            window = "7d"
        else:
            cat = "all"
            window = "24h" if ("today" in q or "24" in q) else "7d"

        min_alert = None
        for level in ALERT_LEVELS:
            if level in q:
                min_alert = level
                break

        return self.get_events(category=cat, window=window, limit=limit, min_alert=min_alert)

    def get_available_feeds(self) -> str:
        """Human-readable catalog of feeds we expose."""
        lines = ["GDACS feeds available via this tool:"]
        for (cat, win), path in sorted(GDACS_FEEDS.items()):
            lines.append(f"  - category='{cat}', window='{win}'  -> {self.base_url}/{path}")
        lines.append("\nAlert levels: green (low), orange (medium), red (high humanitarian impact).")
        return "\n".join(lines)

    @staticmethod
    def _extract(entry, key: str) -> str:
        val = entry.get(key)
        if isinstance(val, dict):
            return val.get("value", "") or val.get("#text", "") or ""
        return str(val) if val is not None else ""

    @staticmethod
    def _extract_alert(entry) -> str:
        for k in ("gdacs_alertlevel", "alertlevel"):
            v = entry.get(k)
            if v:
                return v if isinstance(v, str) else str(v)
        return "unknown"
