from __future__ import annotations
from src.ingest.grants_gov_connector import fetch_grants_gov
from src.ingest.eu_ftop_connector import fetch_eu_ftop
from src.normalize.normalize_opportunities import normalize_opportunities
from src.enrich.agriculture_enrichment import enrich_agriculture

def main() -> None:
    print("Fetching Grants.gov...")
    fetch_grants_gov()

    print("Fetching EU Funding & Tenders...")
    fetch_eu_ftop()

    print("Normalizing...")
    normalize_opportunities(include_world_bank=False)

    print("Applying agriculture enrichment...")
    df = enrich_agriculture()
    print(df["agriculture_flag"].value_counts(dropna=False))
    print("Done.")

if __name__ == "__main__":
    main()
