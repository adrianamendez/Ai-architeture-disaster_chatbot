import logging
import time

import httpx
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NASA_EONET_BASE_URL, HTTP_RETRY

logger = logging.getLogger(__name__)


def _get_with_retry(url: str, params: Optional[dict] = None, timeout: int = 30):
    """GET with exponential backoff. Returns the parsed JSON response or raises."""
    last_exc = None
    for attempt in range(1, HTTP_RETRY["attempts"] + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                logger.debug("GET %s attempt=%d ok status=%d", url, attempt, r.status_code)
                return r.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            last_exc = e
            sleep = HTTP_RETRY["backoff_seconds"] * (2 ** (attempt - 1))
            logger.warning("GET %s attempt=%d/%d failed: %s — backoff %.1fs",
                           url, attempt, HTTP_RETRY["attempts"], e, sleep)
            if attempt < HTTP_RETRY["attempts"]:
                time.sleep(sleep)
    raise last_exc


class NASAEONETTool:
    def __init__(self, base_url: str = NASA_EONET_BASE_URL):
        self.base_url = base_url
        self.categories = {
            "wildfires": "wildfires",
            "severe_storms": "severeStorms",
            "volcanoes": "volcanoes",
            "earthquakes": "earthquakes",
            "floods": "floods",
            "landslides": "landslides",
            "drought": "drought",
            "dust_haze": "dustHaze",
            "sea_lake_ice": "seaLakeIce",
            "snow": "snow",
            "temperature_extremes": "tempExtremes",
            "water_color": "waterColor",
        }

    def get_events(
        self,
        category: Optional[str] = None,
        status: str = "open",
        limit: int = 10,
        days: Optional[int] = None,
    ) -> str:
        params = {"status": status, "limit": limit}
        if days:
            params["days"] = days

        url = f"{self.base_url}/events"
        if category:
            cat_id = self.categories.get(category, category)
            url = f"{self.base_url}/categories/{cat_id}"

        try:
            data = _get_with_retry(url, params=params, timeout=HTTP_RETRY["request_timeout"])
        except httpx.HTTPStatusError as e:
            logger.error("NASA EONET API error: %s", e.response.status_code)
            return f"API error: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error("NASA EONET connection error: %s", e)
            return f"Connection error: {e}"

        events = data.get("events", [])
        if not events:
            return "No active disaster events found for the given criteria."

        results = []
        for event in events[:limit]:
            title = event.get("title", "Unknown")
            event_id = event.get("id", "N/A")
            cats = ", ".join(c.get("title", "") for c in event.get("categories", []))
            sources = ", ".join(s.get("url", "") for s in event.get("sources", []))

            geometries = event.get("geometry", [])
            location = "N/A"
            date = "N/A"
            if geometries:
                latest = geometries[-1]
                coords = latest.get("coordinates", [])
                date = latest.get("date", "N/A")
                if coords:
                    location = f"({coords[1]:.2f}, {coords[0]:.2f})" if len(coords) >= 2 else str(coords)

            results.append(
                f"- **{title}** (ID: {event_id})\n"
                f"  Category: {cats}\n"
                f"  Location: {location}\n"
                f"  Date: {date}\n"
                f"  Sources: {sources}"
            )

        header = f"NASA EONET Events ({status}, limit={limit})"
        if category:
            header += f", category={category}"
        return f"{header}\n\n" + "\n\n".join(results)

    def get_categories(self) -> str:
        try:
            data = _get_with_retry(f"{self.base_url}/categories", timeout=15)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error("NASA EONET categories fetch failed: %s", e)
            return f"Error fetching categories: {e}"

        cats = data.get("categories", [])
        lines = ["Available NASA EONET Categories:"]
        for c in cats:
            lines.append(f"  - {c.get('title', 'Unknown')} (id: {c.get('id', 'N/A')})")
        return "\n".join(lines)

    def search_events(self, query: str, limit: int = 10) -> str:
        query_lower = query.lower()
        matched_category = None
        for key, value in self.categories.items():
            if key.replace("_", " ") in query_lower or value.lower() in query_lower:
                matched_category = key
                break

        status = "open"
        if "closed" in query_lower or "past" in query_lower or "historical" in query_lower:
            status = "closed"
        elif "all" in query_lower:
            status = "all"

        days = None
        for word in query_lower.split():
            if word.isdigit():
                days = int(word)
                break

        return self.get_events(
            category=matched_category,
            status=status,
            limit=limit,
            days=days,
        )
