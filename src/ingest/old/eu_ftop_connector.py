from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.helpers import PROJECT_ROOT, save_parquet_csv

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eu_ftop"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"

# Public portal pages
EU_CALLS_PAGE = "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/calls-for-proposals"

# Optional official search API endpoint if you have the exact URL
EU_FTOP_SEARCH_API_URL = os.getenv("EU_FTOP_SEARCH_API_URL")

@dataclass
class EuFtopIngestConfig:
    max_pages: int = 3
    timeout_seconds: int = 60
    query: str = "agriculture"
    use_api_if_configured: bool = True
    page_size_hint: int = 25

def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)

def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", "null"):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")

def _extract_text(node) -> str | None:
    if node is None:
        return None
    text = node.get_text(" ", strip=True)
    return text or None

def _guess_status(text: str | None) -> str:
    if not text:
        return "unknown"
    t = text.lower()
    if "forthcoming" in t or "upcoming" in t:
        return "forecast"
    if "open" in t:
        return "open"
    if "closed" in t:
        return "closed"
    return "unknown"

def _normalize_listing_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    ingest_ts = datetime.now(timezone.utc).isoformat()
    normalized_rows = []
    for rec in records:
        title = _safe_str(rec.get("title"))
        description = _safe_str(rec.get("description"))
        publish_date = _parse_date(rec.get("publish_date"))
        close_date = _parse_date(rec.get("close_date"))
        keyword_text = " ".join([x for x in [title, description, rec.get("sector_raw")] if x])

        normalized_rows.append({
            "record_id": None,
            "source_system": "eu_ftop",
            "source_record_id": _safe_str(rec.get("source_record_id")),
            "source_url": _safe_str(rec.get("source_url")),
            "title": title,
            "description": description,
            "donor_name": "European Commission",
            "donor_type": "supranational",
            "programme_name": _safe_str(rec.get("programme_name")),
            "opportunity_type": "call_for_proposals",
            "status_raw": _safe_str(rec.get("status_raw")) or "unknown",
            "status_standard": None,
            "country": None,
            "region": "Europe",
            "sector_raw": _safe_str(rec.get("sector_raw")),
            "sector_standard": None,
            "publish_date": publish_date,
            "close_date": close_date,
            "days_to_close": int((close_date - publish_date).days) if not pd.isna(publish_date) and not pd.isna(close_date) else None,
            "keyword_text": keyword_text,
            "agriculture_flag": 0,
            "subtheme": None,
            "is_agriculture_by_sector": 0,
            "is_agriculture_by_text": 0,
            "text_length": len(keyword_text),
            "publish_year": int(publish_date.year) if not pd.isna(publish_date) else None,
            "publish_month": int(publish_date.month) if not pd.isna(publish_date) else None,
            "publish_quarter": int(((publish_date.month - 1)//3)+1) if not pd.isna(publish_date) else None,
            "ingest_timestamp": ingest_ts,
        })
    return pd.DataFrame(normalized_rows)

def _fetch_via_configured_api(config: EuFtopIngestConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    session = requests.Session()

    for page_index in range(1, config.max_pages + 1):
        params = {
            "q": config.query,
            "page": page_index,
            "size": config.page_size_hint,
            "type": "calls-for-proposals",
        }
        response = session.get(EU_FTOP_SEARCH_API_URL, params=params, timeout=config.timeout_seconds)
        response.raise_for_status()
        result = response.json()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = RAW_DIR / f"eu_ftop_api_page_{page_index}_{timestamp}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if isinstance(result, dict):
            batch = []
            for key in ["data", "results", "items", "records"]:
                value = result.get(key)
                if isinstance(value, list):
                    batch.extend([x for x in value if isinstance(x, dict)])
            if not batch and isinstance(result.get("data"), dict):
                for key in ["results", "items", "records"]:
                    value = result["data"].get(key)
                    if isinstance(value, list):
                        batch.extend([x for x in value if isinstance(x, dict)])
        else:
            batch = []

        if not batch:
            break

        for rec in batch:
            records.append({
                "source_record_id": rec.get("id") or rec.get("callId") or rec.get("identifier"),
                "source_url": rec.get("url"),
                "title": rec.get("title"),
                "description": rec.get("description") or rec.get("summary"),
                "programme_name": rec.get("programme") or rec.get("program"),
                "status_raw": rec.get("status"),
                "sector_raw": rec.get("topic") or rec.get("theme"),
                "publish_date": rec.get("publishDate") or rec.get("openingDate"),
                "close_date": rec.get("deadlineDate") or rec.get("closingDate"),
            })

        if len(batch) < config.page_size_hint:
            break

    return records

def _fetch_via_public_pages(config: EuFtopIngestConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 FundingOpportunityResearchBot/1.0"}

    # Public-page fallback. This fetches portal HTML and extracts visible opportunity links.
    for page_index in range(1, config.max_pages + 1):
        page_url = f"{EU_CALLS_PAGE}?text={quote_plus(config.query)}&page={page_index}"
        response = session.get(page_url, timeout=config.timeout_seconds, headers=headers)
        response.raise_for_status()
        html = response.text

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = RAW_DIR / f"eu_ftop_html_page_{page_index}_{timestamp}.html"
        raw_path.write_text(html, encoding="utf-8")

        soup = BeautifulSoup(html, "lxml")
        page_records: list[dict[str, Any]] = []

        # Strategy 1: competitive-calls links
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = _extract_text(a)
            full_url = urljoin("https://ec.europa.eu", href)

            if "opportunities/competitive-calls-cs/" in href or "opportunities/calls-for-proposals" in href:
                key = (href, text)
                if key in seen:
                    continue
                seen.add(key)

                page_records.append({
                    "source_record_id": re.sub(r"\D", "", href.split("/")[-1]) or href,
                    "source_url": full_url,
                    "title": text,
                    "description": None,
                    "programme_name": None,
                    "status_raw": _guess_status(text),
                    "sector_raw": config.query,
                    "publish_date": None,
                    "close_date": None,
                })

        # Strategy 2: parse detail-ish snippets around those links
        # Keep best-effort text extraction from surrounding card/container elements
        enriched = []
        for rec in page_records:
            # find the matching anchor again
            anchor = soup.find("a", href=lambda x: isinstance(x, str) and rec["source_url"].endswith(x) if x else False)
            container = anchor.find_parent(["article", "div", "li"]) if anchor else None
            description = None
            programme = None
            if container:
                text = _extract_text(container)
                if text and rec["title"]:
                    description = text[:2000]
                    # try to extract a simple programme cue
                    m = re.search(r"(Horizon Europe|Erasmus\+|LIFE|Digital Europe|Interreg|EU4Health)", text, flags=re.I)
                    if m:
                        programme = m.group(1)
            rec["description"] = description
            rec["programme_name"] = programme
            enriched.append(rec)

        if not enriched:
            break

        records.extend(enriched)

        # Public pages often paginate irregularly, so stop if very sparse
        if len(enriched) < 5:
            break

    return records

def fetch_eu_ftop(config: EuFtopIngestConfig | None = None) -> pd.DataFrame:
    config = config or EuFtopIngestConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    if EU_FTOP_SEARCH_API_URL and config.use_api_if_configured:
        records = _fetch_via_configured_api(config)
    else:
        records = _fetch_via_public_pages(config)

    df = _normalize_listing_records(records)
    save_parquet_csv(df, STAGING_DIR / "eu_ftop_opportunities.parquet", STAGING_DIR / "eu_ftop_opportunities.csv")
    return df

if __name__ == "__main__":
    df = fetch_eu_ftop()
    print(df.head())
    print(f"Rows: {len(df)}")
