#!/usr/bin/env python3
import sys, time, math, logging, io, os
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
import h3


SR_HEX_PATH = "sr_hex.csv.gz"
SUBURBS_URL = "https://city-of-cape-town-open-data/suburbs.geojson"  # replace with real URL if needed
SUBURB_NAME_COL = "NAME"
REQUEST_ID_COL = "request_id"

WIND_URL = "https://atlantis-air-quality/wind_2020.csv"  # replace with real endpoint
WIND_CACHE = "wind_2020_cache.csv"
WIND_TIME_COL = "timestamp"
WIND_SPEED_COL = "wind_speed"
WIND_DIR_COL = "wind_direction"

RADIUS_M = 1855.0   # ~1 arc-minute of great-circle arc
H3_RES = 8          # ~500–800 m
LOG_FILE = "pipeline.log"

# PII / high-risk columns to separate for manual review if present
PII_COLS = ["resident_name", "phone_number", "free_text"]

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in metres."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))



# Step 1: Suburb centroid & spatial subsample
logging.info("Downloading suburb boundaries...")
suburbs = gpd.read_file(SUBURBS_URL)

cand = suburbs[suburbs[SUBURB_NAME_COL].str.contains("Atlantis", case=False, na=False)]
if cand.empty:
    logging.error("No suburb with name containing 'Atlantis' found.")
    sys.exit(1)

atlantis_suburb = cand.iloc[0]
centroid = atlantis_suburb.geometry.centroid
centroid_lat, centroid_lon = centroid.y, centroid.x
logging.info(f"Centroid of Atlantis suburb: lat={centroid_lat:.6f}, lon={centroid_lon:.6f}")

logging.info("Loading sr_hex dataset...")
sr = pd.read_csv(SR_HEX_PATH)

for col in ["Latitude", "Longitude", "notification_time", REQUEST_ID_COL]:
    if col not in sr.columns:
        logging.error(f"Missing required column '{col}' in {SR_HEX_PATH}")
        sys.exit(1)

def within_radius(lat, lon, c_lat, c_lon, radius_m=RADIUS_M):
    if pd.isna(lat) or pd.isna(lon):
        return False
    d = haversine_m(lat, lon, c_lat, c_lon)
    return d <= radius_m

logging.info("Filtering requests within ~1.85 km of centroid...")
mask = sr.apply(
    lambda r: within_radius(r["Latitude"], r["Longitude"], centroid_lat, centroid_lon),
    axis=1,
)
sr_subsample = sr[mask].copy()
logging.info(f"Subsample size: {len(sr_subsample)} of {len(sr)}")

# Step 2: Download wind data & join
def download_wind_data(url, cache_path=WIND_CACHE, retries=5):
    # Use cache if available
    if os.path.exists(cache_path):
        logging.info(f"Loading cached wind data from {cache_path} ...")
        return pd.read_csv(cache_path)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Downloading wind data (attempt {attempt}) from {url} ...")
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if WIND_TIME_COL not in df.columns:
                raise ValueError(f"Missing '{WIND_TIME_COL}' in wind data.")
            df.to_csv(cache_path, index=False)
            logging.info(f"Downloaded wind data ({len(df)} rows), cached to {cache_path}.")
            return df
        except Exception as e:
            last_err = e
            logging.warning(f"Wind data download failed (attempt {attempt}): {e}")
            time.sleep(2 ** (attempt - 1))

    logging.error("Wind data unavailable after retries; proceeding with NaNs.")
    logging.error(f"Last wind error: {last_err}")
    return pd.DataFrame(columns=[WIND_TIME_COL, WIND_SPEED_COL, WIND_DIR_COL])

wind = download_wind_data(WIND_URL)

# parse times
sr_subsample["notification_time"] = pd.to_datetime(sr_subsample["notification_time"])
if not wind.empty:
    wind[WIND_TIME_COL] = pd.to_datetime(wind[WIND_TIME_COL])

    # sort and nearest-time join
    sr_subsample = sr_subsample.sort_values("notification_time")
    wind = wind.sort_values(WIND_TIME_COL)

    sr_subsample = pd.merge_asof(
        sr_subsample,
        wind,
        left_on="notification_time",
        right_on=WIND_TIME_COL,
        direction="nearest",
    )
else:
    sr_subsample[WIND_SPEED_COL] = pd.NA
    sr_subsample[WIND_DIR_COL] = pd.NA


# Step 3: Anonymisation

def anonymise_location(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    return h3.geo_to_h3(lat, lon, H3_RES)  # ~500–800 m resolution

logging.info("Anonymising locations to H3 and dropping raw lat/lon...")
sr_subsample["h3_index"] = sr_subsample.apply(
    lambda r: anonymise_location(r["Latitude"], r["Longitude"]), axis=1
)
sr_subsample.drop(columns=["Latitude", "Longitude"], inplace=True)

logging.info("Rounding notification_time to 6-hour bins...")
sr_subsample["notification_time"] = sr_subsample["notification_time"].dt.floor("6H")

# Separate PII columns for manual review
present_pii = [c for c in PII_COLS if c in sr_subsample.columns]
if present_pii:
    pii_for_review = sr_subsample[[REQUEST_ID_COL] + present_pii].copy()
    pii_for_review.to_csv("sr_subsample_pii_for_review.csv", index=False)
    sr_subsample.drop(columns=present_pii, inplace=True)
    logging.info(f"Removed PII columns {present_pii} to sr_subsample_pii_for_review.csv")

# Simple k-anonymity check: require at least 2 records per (h3, 6h) cell
logging.info("Applying k-anonymity (k=2) on (h3_index, notification_time)...")
group_sizes = (
    sr_subsample.groupby(["h3_index", "notification_time"])
    .size()
    .rename("group_size")
    .reset_index()
)
sr_subsample = sr_subsample.merge(
    group_sizes, on=["h3_index", "notification_time"], how="left"
)

high_risk = sr_subsample[sr_subsample["group_size"] < 2].copy()
safe = sr_subsample[sr_subsample["group_size"] >= 2].copy()

high_risk.drop(columns=["group_size"], inplace=True)
safe.drop(columns=["group_size"], inplace=True)

if not high_risk.empty:
    high_risk.to_csv("sr_subsample_high_risk_for_review.csv", index=False)
    logging.info(
        f"Moved {len(high_risk)} high-risk rows to sr_subsample_high_risk_for_review.csv "
        "(unique or rare (h3, 6h) cells)."
    )

safe.to_csv("sr_subsample_anonymised.csv", index=False)
logging.info(f"Saved anonymised dataset with {len(safe)} rows to sr_subsample_anonymised.csv")
logging.info("Pipeline completed successfully.")
