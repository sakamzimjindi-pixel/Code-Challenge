#!/usr/bin/env python3
"""
Assign H3 resolution 8 hexagons to service requests and validate against sr_hex.csv.gz.

- Equivalent to joining service requests to city-hex-polygons-8.geojson:
  we compute the H3 index directly from Latitude/Longitude, which is
  precisely how those hex polygons are defined.

- For any request with missing or invalid coordinates, we assign h3_id = 0.

- We validate our assigned h3_id against the expected values in sr_hex.csv.gz
  using request_id as the key.

"""

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager

import pandas as pd
import h3


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def timed_step(step_name: str):
    start = time.time()
    logging.info(f"[START] {step_name} ...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logging.info(f"[END]   {step_name} (took {elapsed:.2f} seconds)")


# Main functions

def assign_h3_index(lat: float, lon: float, resolution: int) -> str:
        """
    Assign a resolution-8 H3 index from latitude/longitude.

    - If lat/lon is missing (NaN) or outside valid ranges, return 0.
    - Otherwise, return the H3 index string.

    This is equivalent to spatially joining the point to the corresponding
    polygon in city-hex-polygons-8.geojson.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "0"
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "0"
    return h3.geo_to_h3(lat, lon, resolution)


def assign_h3_to_requests(df: pd.DataFrame, lat_col: str, lon_col: str,
                          request_id_col: str, resolution: int) -> pd.DataFrame:
        """
    Add an h3_id column to the service requests DataFrame.

    Returns:
        (df_with_h3)
    """
    required_cols = {lat_col, lon_col, request_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["h3_id"] = [
        assign_h3_index(lat, lon, resolution)
        for lat, lon in zip(df[lat_col], df[lon_col])
    ]
    return df


def validate_against_sr_hex(requests_df: pd.DataFrame, sr_hex_df: pd.DataFrame,
                            request_id_col: str, sr_hex_h3_col: str):
        """
    Validate assigned h3_id against sr_hex.csv.gz.

    We:
      - Merge on request_id.
      - Compare requests_df.h3_id vs sr_hex_df[CFG.sr_hex_h3_col].
      - Compute number of mismatches and error rate (percentage) over merged rows.

    Returns:
        (mismatches, total, error_rate_pct)
    """
    merged = requests_df.merge(
        sr_hex_df,
        on=request_id_col,
        how="inner",
        suffixes=("_assigned", "_expected"),
    )
    if merged.empty:
        raise ValueError("No overlapping request_id between datasets.")

    mismatches = (merged["h3_id_assigned"] != merged[f"{sr_hex_h3_col}_expected"]).sum()
    total = len(merged)
    error_rate_pct = (mismatches / total) * 100.0

    logging.info("Validation mismatches: %d / %d (%.2f%%)", mismatches, total, error_rate_pct)
    if mismatches > 0:
        sample = merged.loc[
            merged["h3_id_assigned"] != merged[f"{sr_hex_h3_col}_expected"]
        ].head(10)
        logging.info("Sample mismatches:\n%s", sample)

    return mismatches, total, error_rate_pct


# Main Class

def main():
    parser = argparse.ArgumentParser(description="Assign and validate H3 indices.")
    parser.add_argument("--requests", default="service_requests.csv", help="Service requests CSV")
    parser.add_argument("--sr_hex", default="sr_hex.csv.gz", help="Validation file")
    parser.add_argument("--output", default="service_requests_with_h3.csv", help="Output CSV")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution")
    parser.add_argument("--threshold", type=float,
                        default=float(os.getenv("ERROR_THRESHOLD", 2.0)),
                        help="Error threshold percentage")
    parser.add_argument("--log", default="join_validation.log", help="Log file")
    args = parser.parse_args()

    setup_logging(args.log)
    logging.info("=== H3 Join & Validation Script ===")
    logging.info(f"Requests file: {args.requests}")
    logging.info(f"SR hex file: {args.sr_hex}")
    logging.info(f"H3 resolution: {args.resolution}")
    logging.info(f"Error threshold: {args.threshold:.2f}%")

    t0 = time.time()
    try:
        with timed_step("Load service requests"):
            requests = pd.read_csv(args.requests)

        with timed_step("Assign H3 indices"):
            requests_with_h3 = assign_h3_to_requests(
                requests, "Latitude", "Longitude", "request_id", args.resolution
            )

        with timed_step("Load sr_hex.csv.gz"):
            sr_hex = pd.read_csv(args.sr_hex)

        with timed_step("Validate"):
            mismatches, total, error_rate_pct = validate_against_sr_hex(
                requests_with_h3, sr_hex, "request_id", "h3_id"
            )

        if error_rate_pct > args.threshold:
            logging.error("Error rate %.2f%% exceeds threshold %.2f%%", error_rate_pct, args.threshold)
            requests_with_h3.to_csv("service_requests_with_h3_failed.csv", index=False)
            return 2
		# Save successful output
        with timed_step("Save output"):
            requests_with_h3.to_csv(args.output, index=False)

        logging.info("Total wall-clock time: %.2f seconds", time.time() - t0)
        logging.info("Completed successfully.")
        return 0

    except Exception as e:
        logging.exception("Script failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
















