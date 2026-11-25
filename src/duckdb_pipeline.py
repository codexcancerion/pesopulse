"""
PesoPulse — MIT 118 Final Group Project (Team 3):
Forecasting Philippine Provincial Inflation Using Monthly CPI & Upstream Price Drivers

Data Sources (100% Official Philippine Statistics Authority / OpenSTAT):
  1. Local Class Handout CSV (PSA OpenSTAT 2M/PI/CPI/2018NEW/0012M4ACP22.px):
     Consumer Price Index (2018=100), Jan 2018 - Oct 2026 (42,480 rows x 106 valid months)
  2. 2M/PI/CPI/2018NEW/0012M4ACP23.px : Official CPI YoY Inflation (%) (Jan 2019 - Oct 2026)
  3. 2M/PI/CPI/2018NEW/0012M4ACP25.px : CPI Basket Weights (2018=100) across 120 geolocations
  4. 2M/PI/CPI/2018NEW/0012M4ACP24.px : Purchasing Power of the Peso (2018=100)
  5. 2M/PI/CPI/2018NEW/0012M4ACP31.px : National Core CPI YoY Inflation (%)
  6. 2M/PI/BIH/2018NEW/0022M4ABIR2.px : Bottom 30% Income Households YoY Inflation (%)
  7. 2M/PI/BIH/2018NEW/0022M4ABIR3.px : Bottom 30% Income Households CPI Weights (2018=100)
  8. 2M/PI/WPI/2012/0032M4AWPI1.px    : General Wholesale Price Index (GWPI, 2012=100)
  9. 2M/2018NEW/0042M4ARN01.px        : Retail Prices of Rice (Well-Milled & Regular-Milled)
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FINALS_DIR = BASE_DIR.parent.parent
LOCAL_CPI_CSV = (
    FINALS_DIR
    / "NOV 05"
    / "Consumer Price Index for All Income Households by Commodity Group (2018=100)- January 2018 - October 2026.csv"
)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

PX_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB"

# Exact coordinates (Lat, Lon, Island Group) for all 100 Provinces/HUCs + NCR
PH_GEO_COORDS: dict[str, tuple[float, float, str]] = {
    "National Capital Region (NCR)": (14.60, 120.98, "Luzon"),
    "Abra": (17.59, 120.79, "Luzon"),
    "Benguet": (16.50, 120.66, "Luzon"),
    "Ifugao": (16.83, 121.17, "Luzon"),
    "Kalinga": (17.42, 121.35, "Luzon"),
    "Mountain Province": (17.08, 121.08, "Luzon"),
    "Apayao": (18.02, 121.18, "Luzon"),
    "City of Baguio": (16.41, 120.60, "Luzon"),
    "Ilocos Norte": (18.17, 120.71, "Luzon"),
    "Ilocos Sur": (17.28, 120.47, "Luzon"),
    "La Union": (16.62, 120.32, "Luzon"),
    "Pangasinan": (15.92, 120.33, "Luzon"),
    "Batanes": (20.45, 121.97, "Luzon"),
    "Cagayan": (17.90, 121.80, "Luzon"),
    "Isabela": (16.98, 121.85, "Luzon"),
    "Nueva Vizcaya": (16.33, 121.14, "Luzon"),
    "Quirino": (16.27, 121.53, "Luzon"),
    "Bataan": (14.68, 120.45, "Luzon"),
    "Bulacan": (14.85, 120.88, "Luzon"),
    "Nueva Ecija": (15.58, 121.00, "Luzon"),
    "Pampanga": (15.05, 120.67, "Luzon"),
    "Tarlac": (15.48, 120.60, "Luzon"),
    "Zambales": (15.28, 120.12, "Luzon"),
    "Aurora": (15.77, 121.55, "Luzon"),
    "City of Angeles": (15.15, 120.59, "Luzon"),
    "City of Olongapo": (14.83, 120.28, "Luzon"),
    "Batangas": (13.76, 121.06, "Luzon"),
    "Cavite": (14.28, 120.88, "Luzon"),
    "Laguna": (14.17, 121.33, "Luzon"),
    "Quezon": (14.03, 121.92, "Luzon"),
    "Rizal": (14.58, 121.25, "Luzon"),
    "City of Lucena (Capital)": (13.93, 121.62, "Luzon"),
    "Marinduque": (13.38, 121.95, "Luzon"),
    "Occidental Mindoro": (13.00, 120.92, "Luzon"),
    "Oriental Mindoro": (13.00, 121.40, "Luzon"),
    "Palawan": (9.83, 118.74, "Luzon"),
    "Romblon": (12.50, 122.27, "Luzon"),
    "City of Puerto Princesa (Capital)": (9.74, 118.73, "Luzon"),
    "Albay": (13.18, 123.62, "Luzon"),
    "Camarines Norte": (14.13, 122.77, "Luzon"),
    "Camarines Sur": (13.62, 123.25, "Luzon"),
    "Catanduanes": (13.78, 124.22, "Luzon"),
    "Masbate": (12.22, 123.58, "Luzon"),
    "Sorsogon": (12.92, 124.00, "Luzon"),
    "Aklan": (11.67, 122.33, "Visayas"),
    "Antique": (11.17, 122.07, "Visayas"),
    "Capiz": (11.40, 122.63, "Visayas"),
    "Iloilo": (10.97, 122.57, "Visayas"),
    "Guimaras": (10.57, 122.60, "Visayas"),
    "City of Iloilo (Capital)": (10.72, 122.56, "Visayas"),
    "Bohol": (9.85, 124.14, "Visayas"),
    "Cebu": (10.32, 123.89, "Visayas"),
    "City of Cebu (Capital)": (10.31, 123.89, "Visayas"),
    "City of Lapu-Lapu": (10.31, 123.95, "Visayas"),
    "City of Mandaue": (10.33, 123.94, "Visayas"),
    "Eastern Samar": (11.67, 125.43, "Visayas"),
    "Leyte": (10.85, 124.85, "Visayas"),
    "Northern Samar": (12.42, 124.67, "Visayas"),
    "Samar": (11.83, 125.00, "Visayas"),
    "Southern Leyte": (10.33, 125.08, "Visayas"),
    "Biliran": (11.58, 124.48, "Visayas"),
    "City of Tacloban (Capital)": (11.24, 125.00, "Visayas"),
    "Zamboanga del Norte": (8.15, 123.00, "Mindanao"),
    "Zamboanga del Sur": (7.83, 123.25, "Mindanao"),
    "Zamboanga Sibugay": (7.75, 122.67, "Mindanao"),
    "City of Zamboanga": (6.92, 122.08, "Mindanao"),
    "City of Isabela": (6.70, 121.97, "Mindanao"),
    "Bukidnon": (8.05, 125.00, "Mindanao"),
    "Camiguin": (9.17, 124.73, "Mindanao"),
    "Lanao del Norte": (8.02, 124.02, "Mindanao"),
    "Misamis Occidental": (8.33, 123.75, "Mindanao"),
    "Misamis Oriental": (8.50, 124.75, "Mindanao"),
    "City of Iligan": (8.23, 124.24, "Mindanao"),
    "City of Cagayan de Oro (Capital)": (8.48, 124.65, "Mindanao"),
    "Davao del Norte": (7.55, 125.70, "Mindanao"),
    "Davao del Sur": (6.75, 125.33, "Mindanao"),
    "Davao Oriental": (7.18, 126.33, "Mindanao"),
    "Davao de Oro": (7.60, 126.08, "Mindanao"),
    "Davao Occidental": (6.10, 125.60, "Mindanao"),
    "City of Davao": (7.07, 125.61, "Mindanao"),
    "Cotabato": (7.20, 124.85, "Mindanao"),
    "South Cotabato": (6.33, 124.85, "Mindanao"),
    "Sultan Kudarat": (6.55, 124.28, "Mindanao"),
    "Sarangani": (5.93, 124.98, "Mindanao"),
    "City of General Santos": (6.11, 125.17, "Mindanao"),
    "City of Cotabato": (7.22, 124.25, "Mindanao"),
    "Basilan": (6.58, 122.03, "Mindanao"),
    "Lanao del Sur": (7.82, 124.42, "Mindanao"),
    "Maguindanao": (6.95, 124.42, "Mindanao"),
    "Sulu": (5.98, 121.12, "Mindanao"),
    "Tawi-tawi": (5.13, 120.13, "Mindanao"),
    "Agusan del Norte": (9.00, 125.53, "Mindanao"),
    "Agusan del Sur": (8.55, 125.85, "Mindanao"),
    "Surigao del Norte": (9.68, 125.55, "Mindanao"),
    "Surigao del Sur": (8.67, 126.15, "Mindanao"),
    "Dinagat Islands": (10.13, 125.60, "Mindanao"),
    "City of Butuan (Capital)": (8.95, 125.54, "Mindanao"),
    "Negros Occidental": (10.42, 123.02, "Visayas"),
    "City of Bacolod (Capital)": (10.68, 122.95, "Visayas"),
    "Negros Oriental": (9.62, 123.02, "Visayas"),
    "Siquijor": (9.20, 123.58, "Visayas"),
}

KEY_COICOP_PREFIXES = (
    "0 -",
    "01 -",
    "01.1 -",
    "01.1.1.12 -",
    "02 -",
    "03 -",
    "04 -",
    "04.5 -",
    "05 -",
    "06 -",
    "07 -",
    "07.2.2 -",
    "07.2.2.1 -",
    "07.2.2.2 -",
    "08 -",
    "09 -",
    "10 -",
    "11 -",
    "12 -",
    "13 -",
)

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    "January": 1, "February": 2, "March": 3, "April": 4, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


def _http_json(url: str, payload: dict | None = None) -> dict:
    last_err = None
    for attempt in range(5):
        time.sleep(0.8 + attempt * 1.2)
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            data = None
            if payload is not None:
                headers["Content-Type"] = "application/json"
                data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except Exception as e:
            last_err = e
            print(f"  Retry {attempt + 1}/5 for {url}: {e}")
    raise RuntimeError(f"Failed fetching {url}: {last_err}")


def fetch_pxweb_chunked(
    px_rel_path: str,
    cache_name: str,
    filters: dict | None = None,
    chunk_var: str | None = None,
    chunk_size: int = 2,
) -> pd.DataFrame:
    """
    Fetch a PSA OpenSTAT PxWeb table with optional variable chunking and local CSV caching.
    """
    cache_path = RAW_DIR / cache_name
    if cache_path.exists():
        print(f"  Using cached {cache_name}")
        return pd.read_csv(cache_path)

    url = f"{PX_BASE}/{px_rel_path}"
    print(f"Fetching PSA OpenSTAT metadata: {px_rel_path} ...")
    meta = _http_json(url)
    variables = meta["variables"]

    sel_by_var: dict[str, list[str]] = {}
    maps: list[dict[str, str]] = []
    for var in variables:
        code = var["code"]
        vals = var["values"]
        texts = var["valueTexts"]
        if filters and code in filters:
            fn = filters[code]
            sel_codes = [c for c, t in zip(vals, texts) if fn(c, t)]
        else:
            sel_codes = [c for c in vals if c != ""]
        sel_by_var[code] = sel_codes
        maps.append(dict(zip(vals, texts)))

    var_codes = [v["code"] for v in variables]
    if chunk_var and chunk_var in sel_by_var and len(sel_by_var[chunk_var]) > chunk_size:
        all_vals = sel_by_var[chunk_var]
        chunks = [all_vals[i : i + chunk_size] for i in range(0, len(all_vals), chunk_size)]
    else:
        chunks = [None]

    rows: list[dict] = []
    for idx, ch in enumerate(chunks, 1):
        query_items = []
        for var in variables:
            code = var["code"]
            sel_codes = ch if (ch is not None and code == chunk_var) else sel_by_var[code]
            if len(sel_codes) == len(var["values"]) and "" not in var["values"]:
                query_items.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
            else:
                query_items.append({"code": code, "selection": {"filter": "item", "values": sel_codes}})
        if len(chunks) > 1:
            print(f"    Chunk {idx}/{len(chunks)} for {px_rel_path} ...")
        res = _http_json(url, payload={"query": query_items, "response": {"format": "json"}})
        for item in res.get("data", []):
            row = {vc: maps[i].get(k, k) for i, (vc, k) in enumerate(zip(var_codes, item["key"]))}
            val_raw = item["values"][0] if item.get("values") else None
            row["value_raw"] = val_raw
            rows.append(row)

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    df.to_csv(cache_path, index=False)
    print(f"  Saved {cache_name}: {len(df):,} rows")
    return df


def build_geolocation_dim_from_local_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse exact PSA geolocation hierarchy from file order:
      PHILIPPINES -> ..Region -> ....Province/HUC.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            g = row[0]
            if g not in seen_set:
                seen_set.add(g)
                seen.append(g)

    records = []
    cur_region = "Philippines"
    for raw in seen:
        clean = re.sub(r"^\.+", "", raw).strip()
        if raw == "PHILIPPINES":
            geo_level = "National"
            parent_region = "Philippines"
            is_subnational_leaf = False
        elif raw.startswith("..Areas Outside"):
            geo_level = "Aggregate"
            parent_region = "AONCR"
            is_subnational_leaf = False
        elif raw.startswith("..") and not raw.startswith("...."):
            geo_level = "Region"
            cur_region = clean
            parent_region = clean
            is_subnational_leaf = clean == "National Capital Region (NCR)"
        else:
            geo_level = "Province/HUC"
            parent_region = cur_region
            is_subnational_leaf = True

        lat, lon, island = PH_GEO_COORDS.get(clean, (np.nan, np.nan, _region_to_island(parent_region)))
        records.append({
            "raw_geo": raw,
            "clean_geo": clean,
            "geo_level": geo_level,
            "parent_region": parent_region,
            "island_group": island,
            "latitude": lat,
            "longitude": lon,
            "is_subnational_leaf": is_subnational_leaf,
        })
    return pd.DataFrame(records)


def _region_to_island(region: str) -> str:
    r = region.upper()
    if any(k in r for k in ["NCR", "CAR", "REGION I ", "REGION II ", "REGION III", "CALABARZON", "MIMAROPA", "BICOL", "REGION V"]):
        return "Luzon"
    if any(k in r for k in ["WESTERN VISAYAS", "CENTRAL VISAYAS", "EASTERN VISAYAS", "NEGROS ISLAND", "REGION VI", "REGION VII", "REGION VIII", "NIR"]):
        return "Visayas"
    if any(k in r for k in ["ZAMBOANGA", "NORTHERN MINDANAO", "DAVAO", "SOCCSKSARGEN", "BARMM", "CARAGA", "REGION IX", "REGION X", "REGION XI", "REGION XII", "REGION XIII"]):
        return "Mindanao"
    return "Philippines"


def standardize_short_geo(raw: str) -> str:
    """Normalize Bottom-30% short region codes and HUC/province names to match dim_geolocation."""
    s = re.sub(r"^\.+", "", str(raw)).strip()
    su = s.upper()
    mapping = {
        "NCR": "National Capital Region (NCR)",
        "AONCR": "Areas Outside National Capital Region (AONCR)",
        "CAR": "Cordillera Administrative Region (CAR)",
        "REGION I": "Region I (Ilocos Region)",
        "REGION II": "Region II (Cagayan Valley)",
        "REGION III": "Region III (Central Luzon)",
        "REGION IV-A": "Region IV-A (CALABARZON)",
        "MIMAROPA": "MIMAROPA Region",
        "REGION V": "Region V (Bicol Region)",
        "REGION VI": "Region VI (Western Visayas)",
        "REGION VII": "Region VII (Central Visayas)",
        "REGION VIII": "Region VIII (Eastern Visayas)",
        "REGION IX": "Region IX (Zamboanga Peninsula)",
        "REGION X": "Region X (Northern Mindanao)",
        "REGION XI": "Region XI (Davao Region)",
        "REGION XII": "Region XII (SOCCSKSARGEN)",
        "BARMM": "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)",
        "REGION XIII": "Region XIII (Caraga)",
        "NIR": "Negros Island Region (NIR)",
        "ANGELES CITY": "City of Angeles",
        "BACOLOD CITY": "City of Bacolod (Capital)",
        "BAGUIO CITY": "City of Baguio",
        "BUTUAN CITY": "City of Butuan (Capital)",
        "CAGAYAN DE ORO CITY": "City of Cagayan de Oro (Capital)",
        "CEBU CITY": "City of Cebu (Capital)",
        "COTABATO CITY": "City of Cotabato",
        "DAVAO CITY": "City of Davao",
        "GEN SANTOS CITY": "City of General Santos",
        "ILIGAN CITY": "City of Iligan",
        "ILOILO CITY": "City of Iloilo (Capital)",
        "LAPU-LAPU CITY": "City of Lapu-Lapu",
        "LUCENA CITY": "City of Lucena (Capital)",
        "MANDAUE CITY": "City of Mandaue",
        "OLONGAPO CITY": "City of Olongapo",
        "PUERTO PRINCESA CITY": "City of Puerto Princesa (Capital)",
        "TACLOBAN CITY": "City of Tacloban (Capital)",
        "ZAMBOANGA CITY": "City of Zamboanga",
        "NORTH COTABATO": "Cotabato",
        "WESTERN SAMAR": "Samar",
        "CITY OF ISABELA": "City of Isabela",
    }
    return mapping.get(su, s)


def normalize_rice_geo(raw: str) -> str:
    """Map Retail Rice Geolocation strings to match dim_geolocation.clean_geo."""
    s = re.sub(r"^\.+", "", str(raw)).strip()
    if s.lower() == "philippines":
        return "PHILIPPINES"
    if ("NCR" in s and "National Capital" in s) or s == "City of Manila":
        return "National Capital Region (NCR)"
    rice_map = {
        "City of Bacolod": "City of Bacolod (Capital)",
        "City of Butuan": "City of Butuan (Capital)",
        "City of Cagayan De Oro": "City of Cagayan de Oro (Capital)",
        "City of Cebu": "City of Cebu (Capital)",
        "City of Iloilo": "City of Iloilo (Capital)",
        "City of Lapu-lapu": "City of Lapu-Lapu",
        "City of Lucena": "City of Lucena (Capital)",
        "City of Puerto Princesa": "City of Puerto Princesa (Capital)",
        "City of Tacloban": "City of Tacloban (Capital)",
        "Zamboanga City": "City of Zamboanga",
        "Cotabato (North Cotabato)": "Cotabato",
        "Davao de Oro (Compostela Valley)": "Davao de Oro",
        "Tawi-Tawi": "Tawi-tawi",
    }
    return rice_map.get(s, s)


def is_key_coicop(desc: str) -> bool:
    return any(desc.startswith(p) for p in KEY_COICOP_PREFIXES)


def run_pipeline() -> dict:
    if not LOCAL_CPI_CSV.exists():
        raise FileNotFoundError(f"Expected local CPI CSV not found at: {LOCAL_CPI_CSV}")

    print("=== Step 1: Parsing Geolocation Dimension & Auditing Raw 73.87 MB CPI CSV ===")
    dim_geo = build_geolocation_dim_from_local_csv(LOCAL_CPI_CSV)
    dim_geo.to_csv(DATA_DIR / "dim_geolocation.csv", index=False)
    print(f"  Geolocations mapped: {len(dim_geo)} (Subnational leaves: {int(dim_geo['is_subnational_leaf'].sum())})")

    con = duckdb.connect(":memory:")
    csv_sql = str(LOCAL_CPI_CSV).replace("\\", "/")

    # Inspect valid monthly columns (Jan 2018 - Oct 2026)
    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_csv('{csv_sql}', skip=1, header=true)").df()
    all_cols = cols_df["column_name"].tolist()
    month_cols = [c for c in all_cols if re.match(r"^\d{4}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", c) and c not in ("2026 Nov", "2026 Dec")]
    print(f"  Valid monthly columns detected: {len(month_cols)} ({month_cols[0]} -> {month_cols[-1]})")

    # Audit full 4.5M cell count inside DuckDB before filtering
    month_exprs = ",\n".join([f'TRY_CAST(NULLIF("{c}", \'..\') AS DOUBLE) AS "{c}"' for c in month_cols])
    con.execute(f"""
        CREATE TEMP TABLE raw_cpi_wide AS
        SELECT
            Geolocation AS raw_geo,
            "Commodity Description" AS commodity_desc,
            {month_exprs}
        FROM read_csv('{csv_sql}', skip=1, header=true)
    """)
    raw_row_count = con.execute("SELECT COUNT(*) FROM raw_cpi_wide").fetchone()[0]

    # Unpivot ONLY key COICOP series (All Items + 13 Divisions + Food + Rice + Utilities Fuel + Transport Fuels)
    # across all 120 geolocations x 106 months -> ~254k clean records!
    cols_quoted = ", ".join([f'"{c}"' for c in month_cols])
    key_filter_sql = " OR ".join([f"commodity_desc LIKE '{p}%'" for p in KEY_COICOP_PREFIXES])

    con.register("dim_geo_tbl", dim_geo)
    con.execute(f"""
        CREATE TEMP TABLE cpi_key_long AS
        WITH filtered AS (
            SELECT * FROM raw_cpi_wide WHERE ({key_filter_sql})
        ),
        unpivoted AS (
            UNPIVOT filtered
            ON {cols_quoted}
            INTO NAME period_label VALUE cpi_index
        )
        SELECT
            g.raw_geo,
            g.clean_geo,
            g.geo_level,
            g.parent_region,
            g.island_group,
            g.latitude,
            g.longitude,
            g.is_subnational_leaf,
            u.commodity_desc,
            TRIM(SPLIT_PART(u.commodity_desc, ' - ', 1)) AS coicop_code,
            TRIM(SUBSTRING(u.commodity_desc FROM STRPOS(u.commodity_desc, ' - ') + 3)) AS coicop_name,
            CAST(SPLIT_PART(u.period_label, ' ', 1) AS INTEGER) AS year,
            CASE SPLIT_PART(u.period_label, ' ', 2)
                WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4
                WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8
                WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
            END AS month,
            MAKE_DATE(
                CAST(SPLIT_PART(u.period_label, ' ', 1) AS INTEGER),
                CASE SPLIT_PART(u.period_label, ' ', 2)
                    WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4
                    WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8
                    WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
                END,
                1
            ) AS obs_date,
            u.cpi_index
        FROM unpivoted u
        JOIN dim_geo_tbl g ON u.raw_geo = g.raw_geo
        WHERE u.cpi_index IS NOT NULL
    """)
    cpi_key_count = con.execute("SELECT COUNT(*) FROM cpi_key_long").fetchone()[0]
    print(f"  Unpivoted key COICOP CPI rows: {cpi_key_count:,} (from {raw_row_count * len(month_cols):,} raw cells)")

    print("\n=== Step 2: Fetching Companion PSA OpenSTAT 2M Tables ===")
    # 1. Official CPI Weights (120 x 354 = 42,480)
    df_weights = fetch_pxweb_chunked("2M/PI/CPI/2018NEW/0012M4ACP25.px", "psa_cpi_weights_2018.csv")
    df_weights["clean_geo"] = df_weights["Geolocation"].str.replace(r"^\.+", "", regex=True).str.strip()
    df_weights["coicop_code"] = df_weights["Commodity Description"].str.split(" - ").str[0].str.strip()

    # 2. Official CPI YoY % Changes (key items across 120 geos, chunked by 2 years)
    df_yoy = fetch_pxweb_chunked(
        "2M/PI/CPI/2018NEW/0012M4ACP23.px",
        "psa_cpi_yoy_key_2019_2026.csv",
        filters={
            "Commodity Description": lambda c, t: is_key_coicop(t),
            "Period": lambda c, t: t in MONTH_MAP,
        },
        chunk_var="Year",
        chunk_size=2,
    )
    df_yoy["clean_geo"] = df_yoy["Geolocation"].str.replace(r"^\.+", "", regex=True).str.strip()
    df_yoy["coicop_code"] = df_yoy["Commodity Description"].str.split(" - ").str[0].str.strip()
    df_yoy["year"] = df_yoy["Year"].astype(int)
    df_yoy["month"] = df_yoy["Period"].map(MONTH_MAP)

    # 3. Purchasing Power of Peso (120 geos x 2018-2026)
    df_ppp = fetch_pxweb_chunked(
        "2M/PI/CPI/2018NEW/0012M4ACP24.px",
        "psa_ppp_2018_2026.csv",
        filters={"Period": lambda c, t: t in MONTH_MAP},
    )
    df_ppp["clean_geo"] = df_ppp["Geolocation"].str.replace(r"^\.+", "", regex=True).str.strip()
    df_ppp["year"] = df_ppp["Year"].astype(int)
    df_ppp["month"] = df_ppp["Period"].map(MONTH_MAP)

    # 4. National Core CPI YoY % (key items)
    df_core = fetch_pxweb_chunked(
        "2M/PI/CPI/2018NEW/0012M4ACP31.px",
        "psa_core_cpi_yoy_2019_2026.csv",
        filters={
            "Commodity Description": lambda c, t: is_key_coicop(t),
            "Period": lambda c, t: t in MONTH_MAP,
        },
    )
    df_core["coicop_code"] = df_core["Commodity Description"].str.split(" - ").str[0].str.strip()
    df_core["year"] = df_core["Year"].astype(int)
    df_core["month"] = df_core["Period"].map(MONTH_MAP)

    # 5. Bottom 30% Income Households YoY % & Weights
    b30_prefixes = ("0 -", "01 -", "01.1 -", "01.1.1.12 -", "04 -", "04.5 -", "07 -", "07.2.2 -", "11 -")
    df_b30_yoy = fetch_pxweb_chunked(
        "2M/PI/BIH/2018NEW/0022M4ABIR2.px",
        "psa_b30_yoy_2019_2026.csv",
        filters={
            "Commodity Description": lambda c, t: any(t.startswith(p) for p in b30_prefixes),
            "Period": lambda c, t: t in MONTH_MAP,
        },
        chunk_var="Year",
        chunk_size=2,
    )
    df_b30_yoy["clean_geo"] = df_b30_yoy["Geolocation"].apply(standardize_short_geo)
    df_b30_yoy["coicop_code"] = df_b30_yoy["Commodity Description"].str.split(" - ").str[0].str.strip()
    df_b30_yoy["year"] = df_b30_yoy["Year"].astype(int)
    df_b30_yoy["month"] = df_b30_yoy["Period"].map(MONTH_MAP)

    df_b30_wt = fetch_pxweb_chunked(
        "2M/PI/BIH/2018NEW/0022M4ABIR3.px",
        "psa_b30_weights_2018.csv",
        filters={"Commodity Description": lambda c, t: any(t.startswith(p) for p in b30_prefixes)},
    )
    df_b30_wt["clean_geo"] = df_b30_wt["Geolocation"].apply(standardize_short_geo)
    df_b30_wt["coicop_code"] = df_b30_wt["Commodity Description"].str.split(" - ").str[0].str.strip()

    # 6. General Wholesale Price Index (GWPI, 2018-2026)
    df_gwpi = fetch_pxweb_chunked(
        "2M/PI/WPI/2012/0032M4AWPI1.px",
        "psa_gwpi_2018_2026.csv",
        filters={
            "Commodity Description": lambda c, t: c in ("1", "A", "D"),
            "Year": lambda c, t: int(t) >= 2018,
            "Period": lambda c, t: t in MONTH_MAP,
        },
    )
    df_gwpi["island_group"] = df_gwpi["Geographic location"].str.strip().str.title()
    df_gwpi["gwpi_cat"] = df_gwpi["Commodity Description"].map({
        "ALL ITEMS": "all",
        "..FOOD": "food",
        "..MINERAL FUELS, LUBRICANTS AND RELATED MATERIALS": "fuel",
    })
    df_gwpi["year"] = df_gwpi["Year"].astype(int)
    df_gwpi["month"] = df_gwpi["Period"].map(MONTH_MAP)

    # 7. Retail Prices of Well-Milled & Regular-Milled Rice (2018-2026)
    df_rice = fetch_pxweb_chunked(
        "2M/2018NEW/0042M4ARN01.px",
        "psa_retail_rice_2018_2026.csv",
        filters={
            "Commodity": lambda c, t: c in ("0", "1"),
            "Period": lambda c, t: t in MONTH_MAP,
        },
        chunk_var="Year",
        chunk_size=3,
    )
    df_rice["clean_geo"] = df_rice["Geolocation"].apply(normalize_rice_geo)
    df_rice["rice_type"] = df_rice["Commodity"].map({
        "RICE, WELL-MILLED, 1 KG": "well_milled",
        "RICE, REGULAR-MILLED, 1 KG": "regular_milled",
    })
    df_rice["year"] = df_rice["Year"].astype(int)
    df_rice["month"] = df_rice["Period"].map(MONTH_MAP)

    print("\n=== Step 3: Building Normalized Analytical Marts in DuckDB ===")
    con.register("wt_tbl", df_weights[["clean_geo", "coicop_code", "value"]].rename(columns={"value": "cpi_weight"}))
    con.register("yoy_tbl", df_yoy[["clean_geo", "coicop_code", "year", "month", "value"]].rename(columns={"value": "official_yoy_pct"}))
    con.register("ppp_tbl", df_ppp[["clean_geo", "year", "month", "value"]].rename(columns={"value": "ppp_index"}))
    con.register("core_tbl", df_core[["coicop_code", "year", "month", "value"]].rename(columns={"value": "natl_core_yoy_pct"}))
    con.register("b30_yoy_tbl", df_b30_yoy[["clean_geo", "coicop_code", "year", "month", "value"]].rename(columns={"value": "b30_yoy_pct"}))
    con.register("b30_wt_tbl", df_b30_wt[["clean_geo", "coicop_code", "value"]].rename(columns={"value": "b30_weight"}))
    con.register("gwpi_tbl", df_gwpi[["island_group", "gwpi_cat", "year", "month", "value"]])
    con.register("rice_tbl", df_rice[["clean_geo", "rice_type", "year", "month", "value"]])

    # Save long division-level CPI + weights (within-geolocation normalized %) + YoY + contribution mart
    con.execute("""
        CREATE OR REPLACE TABLE cpi_division_long AS
        WITH wt_norm AS (
            SELECT
                w.clean_geo,
                w.coicop_code,
                w.cpi_weight AS cpi_national_weight,
                ROUND(100.0 * w.cpi_weight / NULLIF(MAX(CASE WHEN w.coicop_code = '0' THEN w.cpi_weight END) OVER (PARTITION BY w.clean_geo), 0), 4) AS cpi_weight
            FROM wt_tbl w
        ),
        b30_wt_norm AS (
            SELECT
                bw.clean_geo,
                bw.coicop_code,
                bw.b30_weight AS b30_national_weight,
                ROUND(100.0 * bw.b30_weight / NULLIF(MAX(CASE WHEN bw.coicop_code = '0' THEN bw.b30_weight END) OVER (PARTITION BY bw.clean_geo), 0), 4) AS b30_weight
            FROM b30_wt_tbl bw
        )
        SELECT
            c.raw_geo,
            c.clean_geo,
            c.geo_level,
            c.parent_region,
            c.island_group,
            c.latitude,
            c.longitude,
            c.is_subnational_leaf,
            c.coicop_code,
            c.coicop_name,
            c.commodity_desc,
            c.year,
            c.month,
            c.obs_date,
            c.cpi_index,
            y.official_yoy_pct,
            w.cpi_weight,
            w.cpi_national_weight,
            ROUND((w.cpi_weight / 100.0) * y.official_yoy_pct, 4) AS weighted_pp_contrib,
            b.b30_yoy_pct,
            bw.b30_weight
        FROM cpi_key_long c
        LEFT JOIN wt_norm w
          ON c.clean_geo = w.clean_geo AND c.coicop_code = w.coicop_code
        LEFT JOIN yoy_tbl y
          ON c.clean_geo = y.clean_geo AND c.coicop_code = y.coicop_code AND c.year = y.year AND c.month = y.month
        LEFT JOIN b30_yoy_tbl b
          ON LOWER(c.clean_geo) = LOWER(b.clean_geo) AND c.coicop_code = b.coicop_code AND c.year = b.year AND c.month = b.month
        LEFT JOIN b30_wt_norm bw
          ON LOWER(c.clean_geo) = LOWER(bw.clean_geo) AND c.coicop_code = bw.coicop_code
    """)
    div_parquet = DATA_DIR / "cpi_division_monthly.parquet"
    con.execute(f"COPY cpi_division_long TO '{str(div_parquet).replace(chr(92), '/')}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # Build GWPI wide + YoY by Island Group
    con.execute("""
        CREATE TEMP TABLE gwpi_yoy AS
        WITH pivoted AS (
            SELECT
                island_group,
                year,
                month,
                MAKE_DATE(year, month, 1) AS obs_date,
                MAX(CASE WHEN gwpi_cat = 'all' THEN value END) AS gwpi_all_idx,
                MAX(CASE WHEN gwpi_cat = 'food' THEN value END) AS gwpi_food_idx,
                MAX(CASE WHEN gwpi_cat = 'fuel' THEN value END) AS gwpi_fuel_idx
            FROM gwpi_tbl
            GROUP BY 1, 2, 3
        )
        SELECT
            island_group,
            year,
            month,
            (gwpi_all_idx / NULLIF(LAG(gwpi_all_idx, 12) OVER (PARTITION BY island_group ORDER BY obs_date), 0) - 1.0) * 100.0 AS gwpi_all_yoy_pct,
            (gwpi_food_idx / NULLIF(LAG(gwpi_food_idx, 12) OVER (PARTITION BY island_group ORDER BY obs_date), 0) - 1.0) * 100.0 AS gwpi_food_yoy_pct,
            (gwpi_fuel_idx / NULLIF(LAG(gwpi_fuel_idx, 12) OVER (PARTITION BY island_group ORDER BY obs_date), 0) - 1.0) * 100.0 AS gwpi_fuel_yoy_pct
        FROM pivoted
    """)

    # Build Retail Rice wide + YoY by Geolocation
    con.execute("""
        CREATE TEMP TABLE rice_yoy AS
        WITH pivoted AS (
            SELECT
                clean_geo,
                year,
                month,
                MAKE_DATE(year, month, 1) AS obs_date,
                AVG(CASE WHEN rice_type = 'well_milled' THEN value END) AS retail_rice_wm_php,
                AVG(CASE WHEN rice_type = 'regular_milled' THEN value END) AS retail_rice_rm_php
            FROM rice_tbl
            GROUP BY 1, 2, 3
        )
        SELECT
            clean_geo,
            year,
            month,
            retail_rice_wm_php,
            retail_rice_rm_php,
            (retail_rice_wm_php / NULLIF(LAG(retail_rice_wm_php, 12) OVER (PARTITION BY clean_geo ORDER BY obs_date), 0) - 1.0) * 100.0 AS retail_rice_wm_yoy_pct
        FROM pivoted
    """)

    # Build Master Subnational Monthly ML Panel (100 Provinces/HUCs + NCR = 101 subnational areas)
    con.execute("""
        CREATE OR REPLACE TABLE cpi_provincial_panel AS
        WITH base AS (
            SELECT
                clean_geo,
                geo_level,
                parent_region,
                island_group,
                latitude,
                longitude,
                year,
                month,
                obs_date,
                MAX(CASE WHEN coicop_code = '0' THEN cpi_index END) AS headline_cpi,
                MAX(CASE WHEN coicop_code = '0' THEN official_yoy_pct END) AS headline_yoy_pct,
                MAX(CASE WHEN coicop_code = '0' THEN b30_yoy_pct END) AS b30_headline_yoy_pct,
                MAX(CASE WHEN coicop_code = '01' THEN official_yoy_pct END) AS food_bev_yoy_pct,
                MAX(CASE WHEN coicop_code = '01' THEN cpi_weight END) AS food_bev_weight,
                MAX(CASE WHEN coicop_code = '01' THEN b30_weight END) AS b30_food_bev_weight,
                MAX(CASE WHEN coicop_code = '01.1.1.12' THEN official_yoy_pct END) AS rice_cpi_yoy_pct,
                MAX(CASE WHEN coicop_code = '01.1.1.12' THEN cpi_weight END) AS rice_weight,
                MAX(CASE WHEN coicop_code = '01.1.1.12' THEN b30_weight END) AS b30_rice_weight,
                MAX(CASE WHEN coicop_code = '04' THEN official_yoy_pct END) AS housing_util_yoy_pct,
                MAX(CASE WHEN coicop_code = '04' THEN cpi_weight END) AS housing_util_weight,
                MAX(CASE WHEN coicop_code = '04.5' THEN official_yoy_pct END) AS elec_gas_fuel_yoy_pct,
                MAX(CASE WHEN coicop_code = '04.5' THEN cpi_weight END) AS elec_gas_fuel_weight,
                MAX(CASE WHEN coicop_code = '07' THEN official_yoy_pct END) AS transport_yoy_pct,
                MAX(CASE WHEN coicop_code = '07' THEN cpi_weight END) AS transport_weight,
                MAX(CASE WHEN coicop_code = '07.2.2' THEN official_yoy_pct END) AS personal_fuel_yoy_pct,
                MAX(CASE WHEN coicop_code = '07.2.2' THEN cpi_weight END) AS personal_fuel_weight,
                MAX(CASE WHEN coicop_code = '07.2.2.1' THEN official_yoy_pct END) AS diesel_yoy_pct,
                MAX(CASE WHEN coicop_code = '07.2.2.2' THEN official_yoy_pct END) AS gasoline_yoy_pct,
                MAX(CASE WHEN coicop_code = '11' THEN official_yoy_pct END) AS restaurants_accomm_yoy_pct,
                MAX(CASE WHEN coicop_code = '11' THEN cpi_weight END) AS restaurants_accomm_weight
            FROM cpi_division_long
            WHERE is_subnational_leaf = TRUE
            GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
        ),
        enriched AS (
            SELECT
                b.*,
                p.ppp_index,
                nc.natl_core_yoy_pct,
                gw.gwpi_all_yoy_pct,
                gw.gwpi_food_yoy_pct,
                gw.gwpi_fuel_yoy_pct,
                rc.retail_rice_wm_php,
                rc.retail_rice_rm_php,
                rc.retail_rice_wm_yoy_pct,
                (b.headline_cpi / NULLIF(LAG(b.headline_cpi, 1) OVER (PARTITION BY b.clean_geo ORDER BY b.obs_date), 0) - 1.0) * 100.0 AS headline_mom_pct,
                (b.b30_headline_yoy_pct - b.headline_yoy_pct) AS b30_inflation_spread_pp,
                CASE WHEN b.headline_yoy_pct > 4.0 THEN 1 ELSE 0 END AS is_breach_bsp_4pct,
                SIN(2.0 * PI() * b.month / 12.0) AS month_sin,
                COS(2.0 * PI() * b.month / 12.0) AS month_cos
            FROM base b
            LEFT JOIN ppp_tbl p
              ON b.clean_geo = p.clean_geo AND b.year = p.year AND b.month = p.month
            LEFT JOIN (SELECT year, month, natl_core_yoy_pct FROM core_tbl WHERE coicop_code = '0') nc
              ON b.year = nc.year AND b.month = nc.month
            LEFT JOIN gwpi_yoy gw
              ON b.island_group = gw.island_group AND b.year = gw.year AND b.month = gw.month
            LEFT JOIN rice_yoy rc
              ON LOWER(b.clean_geo) = LOWER(rc.clean_geo) AND b.year = rc.year AND b.month = rc.month
        ),
        windowed AS (
            SELECT
                *,
                LAG(headline_yoy_pct, 1) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS yoy_lag1,
                LAG(headline_yoy_pct, 2) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS yoy_lag2,
                LAG(headline_yoy_pct, 3) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS yoy_lag3,
                LAG(headline_yoy_pct, 6) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS yoy_lag6,
                LAG(headline_yoy_pct, 12) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS yoy_lag12,
                AVG(headline_yoy_pct) OVER (PARTITION BY clean_geo ORDER BY obs_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS yoy_roll3_mean,
                STDDEV_SAMP(headline_yoy_pct) OVER (PARTITION BY clean_geo ORDER BY obs_date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS yoy_roll6_std,
                LAG(gwpi_all_yoy_pct, 1) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS gwpi_all_yoy_lag1,
                LAG(gwpi_fuel_yoy_pct, 1) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS gwpi_fuel_yoy_lag1,
                LEAD(headline_yoy_pct, 1) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS target_yoy_t1,
                LEAD(headline_yoy_pct, 3) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS target_yoy_t3,
                LEAD(is_breach_bsp_4pct, 1) OVER (PARTITION BY clean_geo ORDER BY obs_date) AS target_breach_4pct_t1
            FROM enriched
        )
        SELECT * FROM windowed
        ORDER BY clean_geo, obs_date
    """)

    panel_df = con.execute("SELECT * FROM cpi_provincial_panel").df()
    panel_csv = DATA_DIR / "cpi_provincial_ml_panel.csv"
    panel_parquet = DATA_DIR / "cpi_provincial_ml_panel.parquet"
    panel_df.to_csv(panel_csv, index=False)
    con.execute(f"COPY cpi_provincial_panel TO '{str(panel_parquet).replace(chr(92), '/')}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # Export clean regional summary CSV for fast validation & reporting
    reg_df = con.execute("""
        SELECT
            clean_geo AS region,
            year,
            month,
            obs_date,
            MAX(CASE WHEN coicop_code = '0' THEN cpi_index END) AS headline_cpi,
            MAX(CASE WHEN coicop_code = '0' THEN official_yoy_pct END) AS headline_yoy_pct,
            MAX(CASE WHEN coicop_code = '0' THEN b30_yoy_pct END) AS b30_yoy_pct,
            MAX(CASE WHEN coicop_code = '01' THEN official_yoy_pct END) AS food_yoy_pct,
            MAX(CASE WHEN coicop_code = '01.1.1.12' THEN official_yoy_pct END) AS rice_yoy_pct,
            MAX(CASE WHEN coicop_code = '04.5' THEN official_yoy_pct END) AS elec_fuel_yoy_pct,
            MAX(CASE WHEN coicop_code = '07' THEN official_yoy_pct END) AS transport_yoy_pct
        FROM cpi_division_long
        WHERE geo_level IN ('National', 'Region', 'Aggregate')
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 4
    """).df()
    reg_df.to_csv(DATA_DIR / "cpi_regional_monthly.csv", index=False)

    audit = {
        "raw_cpi_csv_rows": int(raw_row_count),
        "raw_cpi_monthly_cells": int(raw_row_count * len(month_cols)),
        "cpi_key_long_rows": int(cpi_key_count),
        "provincial_panel_rows": int(len(panel_df)),
        "provincial_panel_yoy_nonnull_rows": int(panel_df["headline_yoy_pct"].notna().sum()),
        "subnational_units": int(panel_df["clean_geo"].nunique()),
        "regions_represented": int(panel_df["parent_region"].nunique()),
        "date_range": f"{panel_df['obs_date'].min()} to {panel_df['obs_date'].max()}",
        "engineered_columns": int(panel_df.shape[1]),
        "missing_retail_rice_wm_rows": int(panel_df["retail_rice_wm_php"].isna().sum()),
        "missing_gwpi_rows": int(panel_df["gwpi_all_yoy_pct"].isna().sum()),
        "missing_b30_rows": int(panel_df["b30_headline_yoy_pct"].isna().sum()),
        "files_generated": [
            "data/dim_geolocation.csv",
            "data/cpi_division_monthly.parquet",
            "data/cpi_provincial_ml_panel.csv",
            "data/cpi_provincial_ml_panel.parquet",
            "data/cpi_regional_monthly.csv",
        ],
    }
    with open(DATA_DIR / "pipeline_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("\n=== Pipeline Complete! Audit Summary ===")
    print(json.dumps(audit, indent=2, default=str))
    return audit


if __name__ == "__main__":
    run_pipeline()
