
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

# Optional project helper import. If unavailable, script still runs standalone.
try:
    from src.utils.helpers import PROJECT_ROOT, save_parquet_csv
except Exception:
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()

    def save_parquet_csv(df: pd.DataFrame, parquet_path: str | Path, csv_path: str | Path | None = None) -> None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        if csv_path:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eu_ftop"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"

EU_FTOP_SEARCH_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
EU_FTOP_FACET_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/facet"

EU_FTOP_API_KEY = os.getenv("EU_FTOP_API_KEY", "SEDIA")
EU_FTOP_FAQ_API_KEY = os.getenv("EU_FTOP_FAQ_API_KEY", "SEDIA_FAQ")


@dataclass
class EuFtopIngestConfig:
    timeout_seconds: int = 60
    language: str = "en"
    max_pages_per_query: int = 50
    page_size_hint: int = 100
    include_tenders: bool = True
    include_grants: bool = True
    include_open: bool = True
    include_closed: bool = True
    include_forthcoming: bool = True
    programme_period: str | None = None
    framework_programme_codes: list[str] | None = None
    search_terms: list[str] = field(default_factory=lambda: [
        "*",
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
    try:
        dt = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(dt):
            return pd.NaT
        return dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(value, errors="coerce")


def _status_codes_from_flags(config: EuFtopIngestConfig) -> list[str]:
    codes = []
    if config.include_open:
        codes.append("31094501")
    if config.include_closed:
        codes.append("31094502")
    if config.include_forthcoming:
        codes.append("31094503")
    return codes


def _type_codes_from_flags(config: EuFtopIngestConfig) -> list[str]:
    """
    Based on the EU F&T API notes:
    - 0 = tenders
    - 1, 2, 8 = grants/calls classes
    """
    codes = []
    if config.include_tenders:
        codes.append("0")
    if config.include_grants:
        codes.extend(["1", "2", "8"])
    return codes


def build_search_query(config: EuFtopIngestConfig) -> dict[str, Any]:
    must_filters: list[dict[str, Any]] = []

    type_codes = _type_codes_from_flags(config)
    if type_codes:
        must_filters.append({"terms": {"type": type_codes}})

    status_codes = _status_codes_from_flags(config)
    if status_codes:
        must_filters.append({"terms": {"status": status_codes}})

    if config.programme_period:
        must_filters.append({"term": {"programmePeriod": config.programme_period}})

    if config.framework_programme_codes:
        must_filters.append({"terms": {"frameworkProgramme": config.framework_programme_codes}})

    return {"bool": {"must": must_filters}}


def _post_search_or_facet(
    url: str,
    api_key: str,
    text: str,
    query: dict[str, Any],
    language: str = "en",
    timeout_seconds: int = 60,
    extra_form_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = {
        "apiKey": api_key,
        "text": text,
        "language": language,
    }

    query_json = json.dumps(query, ensure_ascii=False)

    session = requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": "FundingOpportunityResearchBot/1.0",
    }

    form_fields = {"query": query_json}
    if extra_form_fields:
        form_fields.update({k: str(v) for k, v in extra_form_fields.items()})

    try:
        response = session.post(
            url,
            params=params,
            files={k: (None, v) for k, v in form_fields.items()},
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        response = session.post(
            url,
            params=params,
            data=form_fields,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def fetch_facet_metadata(config: EuFtopIngestConfig | None = None) -> dict[str, Any]:
    config = config or EuFtopIngestConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    query = build_search_query(config)
    result = _post_search_or_facet(
        url=EU_FTOP_FACET_URL,
        api_key=EU_FTOP_API_KEY,
        text="*",
        query=query,
        language=config.language,
        timeout_seconds=config.timeout_seconds,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_DIR / f"eu_ftop_facet_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def _extract_candidate_records(result: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]

    if not isinstance(result, dict):
        return candidates

    for key in ["results", "data", "items", "records", "content", "documents", "hits", "result"]:
        value = result.get(key)
        if isinstance(value, list):
            candidates.extend([x for x in value if isinstance(x, dict)])
        elif isinstance(value, dict):
            for nested_key in ["results", "items", "records", "content", "documents", "hits"]:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    candidates.extend([x for x in nested_value if isinstance(x, dict)])

    if not candidates and any(k in result for k in ["reference", "summary", "metadata", "url"]):
        candidates.append(result)

    return candidates


def _pick_first(rec: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in rec and rec.get(key) not in (None, "", []):
            return rec.get(key)
    return None


def _meta_first(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if isinstance(value, list) and value:
        return value[0]
    return None


def _extract_funding_amount_from_budget_overview(metadata: dict[str, Any]) -> float | None:
    raw = _meta_first(metadata, "budgetOverview")
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        return None

    amounts: list[float] = []
    topic_map = payload.get("budgetTopicActionMap", {})

    if isinstance(topic_map, dict):
        for topic_actions in topic_map.values():
            if not isinstance(topic_actions, list):
                continue
            for action in topic_actions:
                if not isinstance(action, dict):
                    continue
                year_map = action.get("budgetYearMap", {})
                if isinstance(year_map, dict):
                    for value in year_map.values():
                        try:
                            amounts.append(float(value))
                        except Exception:
                            pass

    return max(amounts) if amounts else None


def _record_key(rec: dict[str, Any]) -> str:
    metadata = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
    return (
        _safe_str(_meta_first(metadata, "identifier"))
        or _safe_str(_pick_first(rec, ["reference", "id", "identifier", "url"]))
        or json.dumps(rec, sort_keys=True)
    )


def _normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    ingest_ts = datetime.now(timezone.utc).isoformat()
    normalized_rows: list[dict[str, Any]] = []

    for rec in records:
        metadata = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}

        source_record_id = (
            _meta_first(metadata, "identifier")
            or _pick_first(rec, ["identifier", "id", "callIdentifier", "topicIdentifier", "topicId", "reference"])
        )

        title = (
            _meta_first(metadata, "title")
            or _pick_first(rec, ["title", "summary", "name", "content"])
        )

        description = (
            _meta_first(metadata, "descriptionByte")
            or _pick_first(rec, ["content", "summary", "description", "objective", "teaser"])
        )

        programme_name = (
            _meta_first(metadata, "programmePeriod")
            or _meta_first(metadata, "frameworkProgramme")
            or _pick_first(rec, ["programme", "program"])
        )

        status_raw = (
            _meta_first(metadata, "status")
            or _pick_first(rec, ["status", "statusLabel", "groupById"])
        )

        sector_raw = (
            _meta_first(metadata, "callIdentifier")
            or _meta_first(metadata, "keywords")
            or _meta_first(metadata, "crossCuttingPriorities")
            or _meta_first(metadata, "frameworkProgramme")
        )

        publish_date = _parse_date(
            _meta_first(metadata, "startDate")
            or _pick_first(rec, ["publishDate", "openingDate", "startDate", "publicationDate"])
        )

        close_date = _parse_date(
            _meta_first(metadata, "deadlineDate")
            or _pick_first(rec, ["deadlineDate", "closingDate", "deadlineModel", "endDate"])
        )

        source_url = (
            _meta_first(metadata, "url")
            or _pick_first(rec, ["url", "link"])
        )

        if source_url is None and source_record_id:
            source_url = f"https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{source_record_id}.json"

        funding_amount = _extract_funding_amount_from_budget_overview(metadata)

        keyword_text = " ".join(
            [x for x in [title, description, str(sector_raw) if sector_raw is not None else None] if x]
        )

        normalized_rows.append(
            {
                "record_id": None,
                "source_system": "eu_ftop",
                "source_record_id": _safe_str(source_record_id),
                "source_url": _safe_str(source_url),
                "title": _safe_str(title),
                "description": _safe_str(description),
                "donor_name": "European Commission",
                "donor_type": "supranational",
                "programme_name": _safe_str(programme_name),
                "opportunity_type": "call_for_proposals",
                "status_raw": _safe_str(status_raw) or "unknown",
                "status_standard": None,
                "country": None,
                "region": "Europe",
                "sector_raw": _safe_str(sector_raw),
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
                "funding_amount_eur": funding_amount,
                "ingest_timestamp": ingest_ts,
            }
        )

    return pd.DataFrame(normalized_rows)


def fetch_eu_ftop(config: EuFtopIngestConfig | None = None) -> pd.DataFrame:
    config = config or EuFtopIngestConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for term in config.search_terms:
        query_label = term
        print(f"Fetching EU F&T records for search term: {query_label!r}")

        base_query = build_search_query(config)

        for page_index in range(1, config.max_pages_per_query + 1):
            # The API notes do not clearly document paging, so we try common search paging fields
            # as form-data alongside the documented query payload.
            extra_form_fields = {
                "pageNumber": page_index,
                "pageSize": config.page_size_hint,
            }

            result = _post_search_or_facet(
                url=EU_FTOP_SEARCH_URL,
                api_key=EU_FTOP_API_KEY,
                text=term,
                query=base_query,
                language=config.language,
                timeout_seconds=config.timeout_seconds,
                extra_form_fields=extra_form_fields,
            )

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_term = (query_label or "all").replace(" ", "_").replace("/", "_").replace("*", "all")
            raw_path = RAW_DIR / f"eu_ftop_{safe_term}_page_{page_index}_{timestamp}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            batch = _extract_candidate_records(result)
            if not batch:
                print(f"No records returned for term={query_label!r}, page={page_index}. Stopping this query.")
                break

            new_count = 0
            for rec in batch:
                key = _record_key(rec)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_records.append(rec)
                new_count += 1

            print(
                f"term={query_label!r} page={page_index}: "
                f"raw={len(batch)} new_unique={new_count} total_unique={len(all_records)}"
            )

            if config.max_total_records is not None and len(all_records) >= config.max_total_records:
                print(f"Reached max_total_records={config.max_total_records}. Stopping collection.")
                break

            # Stop if this page returned fewer than the requested page size
            page_size = None
            if isinstance(result, dict):
                for key in ["pageSize", "size"]:
                    if key in result:
                        try:
                            page_size = int(result[key])
                            break
                        except Exception:
                            pass

            effective_page_size = page_size or config.page_size_hint
            if len(batch) < effective_page_size:
                break

            # Also stop if reported total is already reached
            total_count = None
            if isinstance(result, dict):
                for key in ["total", "totalResults", "totalCount", "count"]:
                    if key in result:
                        try:
                            total_count = int(result[key])
                            break
                        except Exception:
                            pass

            if total_count is not None and len(all_records) >= total_count:
                break

            # Defensive stop: if a page adds no new records, paging is probably not advancing
            if new_count == 0:
                print(f"No new unique records for term={query_label!r}, page={page_index}. Stopping this query.")
                break

        if config.max_total_records is not None and len(all_records) >= config.max_total_records:
            break

    df = _normalize_records(all_records)
    save_parquet_csv(
        df,
        STAGING_DIR / "eu_ftop_opportunities.parquet",
        STAGING_DIR / "eu_ftop_opportunities.csv",
    )
    return df


if __name__ == "__main__":
    df = fetch_eu_ftop()
    print(df.head())
    print(f"Rows: {len(df)}")
