from __future__ import annotations
import re
import pandas as pd
from src.utils.helpers import PROJECT_ROOT, save_parquet_csv

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

AGRICULTURE_KEYWORDS = {
    "crop_production": ["crop", "crops", "maize", "cassava", "rice", "wheat", "seed", "harvest", "horticulture"],
    "food_security": ["food security", "nutrition", "hunger", "food systems"],
    "livestock": ["livestock", "dairy", "poultry", "cattle", "goat", "sheep"],
    "irrigation": ["irrigation", "water management", "watershed"],
    "climate_smart_agriculture": ["climate-smart agriculture", "agroecology", "sustainable agriculture"],
    "rural_livelihoods": ["rural livelihoods", "smallholder", "smallholders", "rural development"],
}

SECTOR_MAP = {
    "agriculture": ["agriculture", "agricultural", "farm", "farming", "agribusiness", "crop", "livestock"],
    "food_security": ["food security", "nutrition", "food systems", "hunger"],
    "climate_environment": ["climate", "environment", "resilience", "adaptation", "sustainability"],
    "rural_development": ["rural", "smallholder", "livelihood"],
    "water_irrigation": ["water", "irrigation", "watershed"],
    "research_innovation": ["research", "innovation", "technology", "science"],
}

def _normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def map_sector_standard(sector_raw: str, keyword_text: str) -> str:
    text = f"{_normalize_text(sector_raw)} {_normalize_text(keyword_text)}"
    for sector, terms in SECTOR_MAP.items():
        if any(term in text for term in terms):
            return sector
    return "other"

def detect_subtheme(keyword_text: str) -> str:
    text = _normalize_text(keyword_text)
    for subtheme, terms in AGRICULTURE_KEYWORDS.items():
        if any(term in text for term in terms):
            return subtheme
    return "other"

def enrich_agriculture() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "opportunities.parquet").copy()
    df["sector_standard"] = df.apply(lambda r: map_sector_standard(r.get("sector_raw", ""), r.get("keyword_text", "")), axis=1)
    df["subtheme"] = df["keyword_text"].apply(detect_subtheme)
    df["is_agriculture_by_sector"] = df["sector_standard"].isin(["agriculture", "food_security", "rural_development", "water_irrigation"]).astype(int)
    df["is_agriculture_by_text"] = df["keyword_text"].fillna("").str.lower().apply(
        lambda x: 1 if any(word in x for word in ["agriculture", "food", "farm", "crop", "rural", "livelihood", "irrigation", "seed", "livestock"]) else 0
    )
    df["agriculture_flag"] = ((df["is_agriculture_by_sector"] == 1) | (df["is_agriculture_by_text"] == 1)).astype(int)
    df["text_length"] = df["keyword_text"].fillna("").str.len()
    save_parquet_csv(df, FEATURES_DIR / "classification_dataset.parquet", FEATURES_DIR / "classification_dataset.csv")
    return df

if __name__ == "__main__":
    df = enrich_agriculture()
    print(df["agriculture_flag"].value_counts(dropna=False))
