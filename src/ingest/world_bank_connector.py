from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.utils.helpers import PROJECT_ROOT, save_parquet_csv

WORLD_BANK_API_URL = "https://search.worldbank.org/api/v2/projects"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "world_bank"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"

@dataclass
class WorldBankIngestConfig:
    rows_per_page: int = 200
    max_pages: int = 3
    timeout_seconds: int = 60

def _safe_get(record: dict[str, Any], key: str, default: Any = None) -> Any:
    value = record.get(key, default)
    if isinstance(value, str):
        value = value.strip()
    return value

def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", "null"):
        return pd.NaT
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return pd.NaT
    return dt.tz_localize(None)

def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    projects = payload.get("projects", {})
    if not isinstance(projects, dict):
        return []
    return list(projects.values())

def fetch_world_bank_context(config: WorldBankIngestConfig | None = None) -> pd.DataFrame:
    config = config or WorldBankIngestConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_records = []

    for page in range(0, config.max_pages):
        params = {"format": "json", "rows": config.rows_per_page, "os": page * config.rows_per_page}
        response = session.get(WORLD_BANK_API_URL, params=params, timeout=config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = RAW_DIR / f"world_bank_projects_page_{page}_{timestamp}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        batch = _extract_records(payload)
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < config.rows_per_page:
            break

    ingest_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for rec in all_records:
        title = _safe_get(rec, "project_name") or _safe_get(rec, "projectname")
        description = _safe_get(rec, "development_objective") or _safe_get(rec, "project_abstract") or _safe_get(rec, "abstract")
        publish_date = _parse_date(_safe_get(rec, "boardapprovaldate"))
        close_date = _parse_date(_safe_get(rec, "closingdate"))
        keyword_text = " ".join([x for x in [title, description] if x])

        rows.append({
            "record_id": None,
            "source_system": "world_bank",
            "source_record_id": _safe_get(rec, "id") or _safe_get(rec, "projectid"),
            "source_url": None,
            "title": title,
            "description": description,
            "donor_name": "World Bank",
            "donor_type": "multilateral",
            "programme_name": None,
            "opportunity_type": "mixed_status_project_context",
            "status_raw": _safe_get(rec, "status"),
            "status_standard": None,
            "country": _safe_get(rec, "countryname"),
            "region": _safe_get(rec, "regionname"),
            "sector_raw": _safe_get(rec, "sector1") or _safe_get(rec, "majorsector_percent"),
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
    df = pd.DataFrame(rows)
    save_parquet_csv(df, STAGING_DIR / "world_bank_context.parquet", STAGING_DIR / "world_bank_context.csv")
    return df

if __name__ == "__main__":
    df = fetch_world_bank_context()
    print(df.head())
    print(f"Rows: {len(df)}")
