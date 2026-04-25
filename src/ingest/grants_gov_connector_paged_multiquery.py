
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.utils.helpers import PROJECT_ROOT, save_parquet_csv

GRANTS_SEARCH_URL = "https://api.grants.gov/v1/api/search2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "grants_gov"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"


@dataclass
class GrantsGovIngestConfig:
    rows: int = 100
    max_pages_per_query: int = 100
    timeout_seconds: int = 60
    opportunity_status: str = ""
    search_terms: list[str] = field(default_factory=lambda: [
        "",
        "agriculture",
        "food",
        "nutrition",
        "crop",
        "livestock",
        "irrigation",
        "soil",
        "climate",
        "sustainability",
        "biodiversity",
        "rural development",
        "smallholder",
        "farm",
        "farmer",
        "seed",
        "fertilizer",
        "water",
    ])
    max_total_records: int | None = 20000


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


def extract_opportunity_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(result, dict):
        return candidates

    for key in ["data", "oppHits", "opportunities", "results", "items"]:
        value = result.get(key)
        if isinstance(value, list):
            candidates.extend([x for x in value if isinstance(x, dict)])

    data_value = result.get("data")
    if not candidates and isinstance(data_value, dict):
        for key in ["oppHits", "results", "items", "opportunities"]:
            value = data_value.get(key)
            if isinstance(value, list):
                candidates.extend([x for x in value if isinstance(x, dict)])

    return candidates


def _record_key(rec: dict[str, Any]) -> str:
    return (
        _safe_str(rec.get("opportunityId"))
        or _safe_str(rec.get("opportunityNumber"))
        or _safe_str(rec.get("id"))
        or json.dumps(rec, sort_keys=True)
    )


def fetch_grants_gov(config: GrantsGovIngestConfig | None = None) -> pd.DataFrame:
    config = config or GrantsGovIngestConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for term in config.search_terms:
        query_label = term if term else "all"
        print(f"Fetching Grants.gov records for search term: {query_label!r}")

        for page_index in range(1, config.max_pages_per_query + 1):
            payload = {
                "rows": config.rows,
                "startRecordNum": ((page_index - 1) * config.rows) + 1,
                "keyword": term,
                "oppStatuses": [config.opportunity_status] if config.opportunity_status else [],
                "sortBy": "openDate|desc",
            }

            response = session.post(
                GRANTS_SEARCH_URL,
                json=payload,
                timeout=config.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_term = query_label.replace(" ", "_").replace("/", "_")
            raw_path = RAW_DIR / f"grants_gov_{safe_term}_page_{page_index}_{timestamp}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            opps = extract_opportunity_records(result)
            if not opps:
                print(f"No records returned for term={query_label!r}, page={page_index}. Stopping this query.")
                break

            new_count = 0
            for rec in opps:
                key = _record_key(rec)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_rows.append(rec)
                new_count += 1

            print(
                f"term={query_label!r} page={page_index}: "
                f"raw={len(opps)} new_unique={new_count} total_unique={len(all_rows)}"
            )

            if config.max_total_records is not None and len(all_rows) >= config.max_total_records:
                print(f"Reached max_total_records={config.max_total_records}. Stopping collection.")
                break

            if len(opps) < config.rows:
                break

        if config.max_total_records is not None and len(all_rows) >= config.max_total_records:
            break

    ingest_ts = datetime.now(timezone.utc).isoformat()
    normalized_rows = []

    for rec in all_rows:
        title = _safe_str(rec.get("opportunityTitle")) or _safe_str(rec.get("title"))
        description = _safe_str(rec.get("synopsis")) or _safe_str(rec.get("description"))
        donor_name = _safe_str(rec.get("agency")) or _safe_str(rec.get("agencyName")) or "Unknown Agency"
        publish_date = _parse_date(rec.get("openDate") or rec.get("postDate"))
        close_date = _parse_date(rec.get("closeDate"))
        category = _safe_str(rec.get("fundingCategory")) or _safe_str(rec.get("category"))
        status_raw = _safe_str(rec.get("oppStatus")) or _safe_str(rec.get("status")) or "unknown"
        keyword_text = " ".join([x for x in [title, description, category] if x])

        normalized_rows.append({
            "record_id": None,
            "source_system": "grants_gov",
            "source_record_id": _safe_str(rec.get("opportunityId")) or _safe_str(rec.get("opportunityNumber")) or _safe_str(rec.get("id")),
            "source_url": f"https://www.grants.gov/search-results-detail/{rec.get('opportunityNumber')}" if rec.get("opportunityNumber") else None,
            "title": title,
            "description": description,
            "donor_name": donor_name,
            "donor_type": "government",
            "programme_name": None,
            "opportunity_type": "grant_opportunity",
            "status_raw": status_raw,
            "status_standard": None,
            "country": "United States",
            "region": "North America",
            "sector_raw": category,
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
            "publish_quarter": int(((publish_date.month - 1) // 3) + 1) if not pd.isna(publish_date) else None,
            "ingest_timestamp": ingest_ts,
        })

    df = pd.DataFrame(normalized_rows)
    save_parquet_csv(
        df,
        STAGING_DIR / "grants_gov_opportunities.parquet",
        STAGING_DIR / "grants_gov_opportunities.csv",
    )
    return df


if __name__ == "__main__":
    df = fetch_grants_gov()
    print(df.head())
    print(f"Rows: {len(df)}")
