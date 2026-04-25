from __future__ import annotations
import pandas as pd
from src.utils.helpers import PROJECT_ROOT, save_parquet_csv

STAGING_DIR = PROJECT_ROOT / "data" / "staging"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATUS_MAP = {
    "forecast": "forecast",
    "pipeline": "pipeline",
    "open": "open",
    "posted": "open",
    "closed": "closed",
    "active": "active",
    "completed": "completed",
}

def normalize_status(value: str | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).lower().strip()
    for key, mapped in STATUS_MAP.items():
        if key in text:
            return mapped
    return "unknown"

def normalize_opportunities(include_world_bank: bool = False) -> pd.DataFrame:
    frames = []
    gg_path = STAGING_DIR / "grants_gov_opportunities.parquet"
    eu_path = STAGING_DIR / "eu_ftop_opportunities.parquet"
    wb_path = STAGING_DIR / "world_bank_context.parquet"

    if gg_path.exists():
        frames.append(pd.read_parquet(gg_path))
    if eu_path.exists():
        frames.append(pd.read_parquet(eu_path))
    if include_world_bank and wb_path.exists():
        frames.append(pd.read_parquet(wb_path))

    if not frames:
        raise FileNotFoundError("No staged source files found.")

    df = pd.concat(frames, ignore_index=True)
    df["status_standard"] = df["status_raw"].apply(normalize_status)
    df["record_id"] = df["source_system"].fillna("unknown").astype(str) + "_" + df["source_record_id"].fillna("missing").astype(str)
    save_parquet_csv(df, PROCESSED_DIR / "opportunities.parquet", PROCESSED_DIR / "opportunities.csv")
    return df

if __name__ == "__main__":
    df = normalize_opportunities(include_world_bank=False)
    print(df.head())
    print(f"Rows: {len(df)}")
