#!/usr/bin/env python3
"""
Read H3 resolution 8 hex polygons from city-hex-polygons-8-10.geojson
using AWS S3 Select, validate against city-hex-polygons-8.geojson,
and compute a schema conformance score based on hex_schema.json.

Major steps:
  1. S3 Select: stream only resolution = 8 features from the 8–10 file.
  2. Validation A (content): compare against the dedicated 8-only file.
  3. Validation B (schema): compute a conformance score using hex_schema.json.
  4. Log timings and metrics; fail if thresholds are exceeded.

This is designed to:
  - Minimise data transferred from S3 (filter on the server via S3 Select).
  - Avoid loading unnecessary resolutions (9, 10) client-side.
  - Provide explicit, quantitative validation of both content and schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import boto3
from botocore.config import Config as BotoConfig
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass
class AppConfig:
    # S3 location of the 8–10 GeoJSON
    bucket_8_10: str = "s3://cct-ds-code-challenge-input-data/"
    key_8_10: str = "city-hex-polygons-8-10.geojson"

    # Validation 8-only GeoJSON. You can use S3 (get_object), S3 Select, or local.
    s3_validation: bool = True
    validation_use_select: bool = False  # if True, use S3 Select on the 8-only file too
    bucket_8: str = "s3://cct-ds-code-challenge-input-data/"
    key_8: str = "city-hex-polygons-8.geojson"
    local_8_path: str = "city-hex-polygons-8.geojson"  # used if s3_validation == False

    # Schema config
    schema_path: str = "hex_schema.json"

    # Logging / outputs
    log_file: str = "s3_select_hex_res8.log"
    output_csv: str = "city_hex_res8_from_s3select.csv"
    schema_metrics_json: str = "schema_metrics.json"
    missing_h3_file: str = "missing_h3index.txt"
    extra_h3_file: str = "extra_h3index.txt"

    # Thresholds
    content_threshold_pct: float = 99.0
    # schema threshold is read from hex_schema.json ("scoring.threshold_pct")

    # AWS client config
    aws_region: Optional[str] = None  # e.g. "eu-west-1" (falls back to env/defaults)
    aws_endpoint_url: Optional[str] = None  # custom endpoint if needed
    max_attempts: int = 5
    connect_timeout: int = 10
    read_timeout: int = 60

    # Compression handling for S3 Select input. One of: "auto", "none", "gzip"
    # "auto": infer gzip if key endswith .gz or .gzip
    # Note: S3 Select supports GZIP and BZIP2 for JSON.
    compression: str = "auto"

    # Developer convenience
    sample_limit: Optional[int] = None  # limit rows for quick test (applies to S3 Select results only)


CFG = AppConfig()

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(CFG.log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@contextmanager
def timed_step(name: str):
    start = time.time()
    logging.info(f"[START] {name} ...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logging.info(f"[END]   {name} (took {elapsed:.2f} seconds)")


def _infer_compression_from_key(key: str, cfg_value: str) -> Optional[str]:
    v = (cfg_value or "none").lower()
    if v == "none":
        return None
    if v == "gzip":
        return "GZIP"
    if v == "auto":
        if key.lower().endswith((".gz", ".gzip")):
            return "GZIP"
        return None
    # fallback
    return None


def _make_s3_client() -> Any:
    boto_cfg = BotoConfig(
        retries={"max_attempts": CFG.max_attempts, "mode": "standard"},
        connect_timeout=CFG.connect_timeout,
        read_timeout=CFG.read_timeout,
    )
    session = boto3.session.Session(region_name=CFG.aws_region)
    return session.client("s3", endpoint_url=CFG.aws_endpoint_url, config=boto_cfg)


def s3_select_json_rows(
    bucket: str,
    key: str,
    sql: str,
) -> List[Dict[str, Any]]:
    """
    Generic S3 Select for JSON documents returning line-delimited JSON objects.

    Returns a list of dicts (rows). You can limit rows via CFG.sample_limit.
    """
    client = _make_s3_client()
    compression_type = _infer_compression_from_key(key, CFG.compression)

    input_ser = {"JSON": {"Type": "DOCUMENT"}}
    if compression_type:
        input_ser["CompressionType"] = compression_type

    logging.info(
        "Running S3 Select on s3://%s/%s (compression=%s) ...",
        bucket, key, compression_type or "none"
    )

    response = client.select_object_content(
        Bucket=bucket,
        Key=key,
        ExpressionType="SQL",
        Expression=sql,
        InputSerialization=input_ser,
        OutputSerialization={"JSON": {"RecordDelimiter": "\n"}},
    )

    rows: List[Dict[str, Any]] = []
    total_bytes = {"scanned": 0, "processed": 0, "returned": 0}

    collected = 0
    for event in response["Payload"]:
        if "Records" in event:
            payload = event["Records"]["Payload"].decode("utf-8")
            for line in payload.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows.append(rec)
                collected += 1
                if CFG.sample_limit and collected >= CFG.sample_limit:
                    break
            if CFG.sample_limit and collected >= CFG.sample_limit:
                break
        elif "Stats" in event:
            stats = event["Stats"]["Details"]
            total_bytes["scanned"] += stats.get("BytesScanned", 0)
            total_bytes["processed"] += stats.get("BytesProcessed", 0)
            total_bytes["returned"] += stats.get("BytesReturned", 0)
        elif "End" in event:
            break

    logging.info(
        "S3 Select stats: BytesScanned=%d, BytesProcessed=%d, BytesReturned=%d",
        total_bytes["scanned"], total_bytes["processed"], total_bytes["returned"]
    )
    logging.info("S3 Select returned %d records.", len(rows))
    return rows


# -------------------------------------------------------------------
# S3 Select: read only resolution-8 features from 8–10
# -------------------------------------------------------------------

def s3_select_res8_hexes() -> pd.DataFrame:
    """
    Use S3 Select to read only resolution=8 features from city-hex-polygons-8-10.geojson.
    We DO NOT pull geometry coordinates to keep transfer small.
    """
    sql = """
        SELECT
          s.id,
          s.properties.resolution AS resolution,
          s.properties.h3index AS h3index,
          s.properties.area_m2 AS area_m2,
          s.geometry.type AS geometry_type
        FROM S3Object[*].features[*] s
        WHERE s.properties.resolution = 8
    """
    records = s3_select_json_rows(CFG.bucket_8_10, CFG.key_8_10, sql)
    df = pd.DataFrame.from_records(records, columns=[
        "id", "resolution", "h3index", "area_m2", "geometry_type"
    ])

    # Normalize & sanity-check
    if "h3index" in df.columns:
        df["h3index"] = df["h3index"].astype(str).str.strip()
    if "geometry_type" in df.columns:
        df["geometry_type"] = df["geometry_type"].astype(str).str.strip()

    # Remove duplicates by h3index (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["h3index"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logging.warning("Removed %d duplicate h3index rows from S3 Select result.", removed)

    # Sanity-check: all resolution=8 (some datasets might be inconsistent)
    if "resolution" in df.columns:
        bad = df[~df["resolution"].isna() & (df["resolution"] != 8)]
        if not bad.empty:
            logging.warning("Found %d rows where resolution != 8 (unexpected).", len(bad))

    logging.info("Final S3 Select res=8 DataFrame rows=%d", len(df))
    return df


# -------------------------------------------------------------------
# Load validation GeoJSON (8-only)
# -------------------------------------------------------------------

def load_validation_hexes() -> pd.DataFrame:
    if CFG.s3_validation:
        if CFG.validation_use_select:
            # Use S3 Select to extract only the needed fields from the 8-only file.
            # This reduces transfer if the file includes large geometries.
            logging.info(
                "Validation via S3 Select from s3://%s/%s ...",
                CFG.bucket_8, CFG.key_8
            )
            sql = """
                SELECT
                    s.id,
                    s.properties.resolution AS resolution,
                    s.properties.h3index AS h3index,
                    s.properties.area_m2 AS area_m2,
                    s.geometry.type AS geometry_type
                FROM S3Object[*].features[*] s
            """
            records = s3_select_json_rows(CFG.bucket_8, CFG.key_8, sql)
            df = pd.DataFrame.from_records(records, columns=[
                "id", "resolution", "h3index", "area_m2", "geometry_type"
            ])
        else:
            logging.info(
                "Loading validation GeoJSON via GetObject from s3://%s/%s ...",
                CFG.bucket_8, CFG.key_8
            )
            s3 = _make_s3_client()
            obj = s3.get_object(Bucket=CFG.bucket_8, Key=CFG.key_8)
            data = obj["Body"].read().decode("utf-8")
            geo = json.loads(data)
            features = geo.get("features", [])
            rows = []
            for f in features:
                props = f.get("properties", {}) or {}
                rows.append({
                    "id": f.get("id"),
                    "resolution": props.get("resolution"),
                    "h3index": props.get("h3index"),
                    "area_m2": props.get("area_m2"),
                    "geometry_type": (f.get("geometry") or {}).get("type"),
                })
            df = pd.DataFrame(rows)
    else:
        logging.info("Loading validation GeoJSON from local file %s ...", CFG.local_8_path)
        with open(CFG.local_8_path, "r", encoding="utf-8") as f:
            geo = json.load(f)
        features = geo.get("features", [])
        rows = []
        for f in features:
            props = f.get("properties", {}) or {}
            rows.append({
                "id": f.get("id"),
                "resolution": props.get("resolution"),
                "h3index": props.get("h3index"),
                "area_m2": props.get("area_m2"),
                "geometry_type": (f.get("geometry") or {}).get("type"),
            })
        df = pd.DataFrame(rows)

    if "h3index" in df.columns:
        df["h3index"] = df["h3index"].astype(str).str.strip()
    if "geometry_type" in df.columns:
        df["geometry_type"] = df["geometry_type"].astype(str).str.strip()

    # Filter resolution=8 only if present and not already guaranteed
    if "resolution" in df.columns:
        res_ct = df["resolution"].dropna().unique().tolist()
        if any(v != 8 for v in res_ct if v is not None):
            logging.warning("Validation file contains non-8 resolution rows; filtering to res=8.")
            df = df[df["resolution"] == 8].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["h3index"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logging.warning("Removed %d duplicate h3index rows from validation set.", removed)

    logging.info("Validation 8-only DataFrame rows=%d", len(df))
    return df


# -------------------------------------------------------------------
# Content validation vs 8-only file
# -------------------------------------------------------------------

def content_validation(res8_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[float, int, int, int]:
    """
    Compare the res=8 subset from 8–10 file vs the dedicated 8-only file.

    We:
      - Compare sets of h3index values.
      - Compute coverage: what % of validation h3index values appear in res8_df.
      - Also report extras present in res8_df but not in validation.

    Returns:
      (coverage_pct, missing_in_res8_count, total_validation, extras_count)
    """
    s_res8 = set(res8_df["h3index"].dropna().astype(str))
    s_val = set(val_df["h3index"].dropna().astype(str))

    missing = s_val - s_res8
    extras = s_res8 - s_val

    coverage = 0.0 if not s_val else (len(s_val) - len(missing)) / len(s_val) * 100.0
    logging.info(
        "Content validation: %.2f%% of validation h3index present in S3 Select result.",
        coverage,
    )
    logging.info(
        "Validation set size=%d, missing_in_res8=%d, extras_in_res8=%d",
        len(s_val), len(missing), len(extras)
    )

    # Persist lists for debugging
    if missing:
        with open(CFG.missing_h3_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(missing)))
        logging.info("Wrote missing h3index list to %s", CFG.missing_h3_file)

    if extras:
        with open(CFG.extra_h3_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(extras)))
        logging.info("Wrote extra h3index list to %s", CFG.extra_h3_file)

    return coverage, len(missing), len(s_val), len(extras)


# -------------------------------------------------------------------
# Schema validation & conformance score (granular)
# -------------------------------------------------------------------

def load_schema() -> Dict[str, Any]:
    with open(CFG.schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_type(value: Any, expected_type: str) -> bool:
    if value is None:
        return False
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "int":
        # allow ints-ish
        return isinstance(value, int) or (isinstance(value, float) and float(value).is_integer())
    if expected_type == "number":
        return isinstance(value, (int, float))
    if expected_type == "array":
        return isinstance(value, (list, tuple))
    if expected_type == "boolean":
        return isinstance(value, bool)
    # default: treat as pass
    return True


def schema_conformance_metrics(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce granular metrics:
      - per-field presence rate
      - per-field type correctness rate (among present)
      - per-field constraint compliance rate (among present)
      - overall score using schema["scoring"]["weights"] (required_presence, type_correctness, constraint_compliance)
      - threshold from schema["scoring"]["threshold_pct"]
    """
    required: Dict[str, str] = schema.get("required_fields", {})
    constraints: Dict[str, Dict[str, Any]] = schema.get("constraints", {})
    scoring_cfg = schema.get("scoring", {}) or {}
    weights = scoring_cfg.get(
        "weights",
        {"required_presence": 0.5, "type_correctness": 0.25, "constraint_compliance": 0.25},
    )
    threshold = scoring_cfg.get("threshold_pct", 95.0)

    metrics = {
        "row_count": int(len(df)),
        "weights": weights,
        "threshold_pct": float(threshold),
        "fields": {},
    }

    if df.empty:
        return {
            **metrics,
            "overall_score_pct": 0.0,
            "note": "No records to score."
        }

    # Ensure all required columns exist to avoid KeyErrors
    for col in set(list(required.keys()) + list(constraints.keys())):
        if col not in df.columns:
            df[col] = None  # missing column entirely

    field_scores = []
    for field, ftype in required.items():
        present_mask = df[field].notna()
        present_count = int(present_mask.sum())
        presence_rate = present_count / len(df) if len(df) else 0.0

        # Among present, check type
        type_ok_count = 0
        constraint_ok_count = 0

        if present_count > 0:
            # Type correctness
            for val in df.loc[present_mask, field]:
                if _check_type(val, ftype):
                    type_ok_count += 1

            # Constraint compliance
            rules = constraints.get(field, {})
            if rules:
                for val in df.loc[present_mask, field]:
                    ok = True
                    # enums
                    if "enum" in rules:
                        ok = ok and (val in rules["enum"])
                    # pattern
                    if ok and "pattern" in rules and isinstance(val, str):
                        import re
                        if not re.match(rules["pattern"], val):
                            ok = False
                    # numeric bounds
                    if ok and "min" in rules and isinstance(val, (int, float)):
                        if val < rules["min"]:
                            ok = False
                    if ok and "max" in rules and isinstance(val, (int, float)):
                        if val > rules["max"]:
                            ok = False

                    if ok:
                        constraint_ok_count += 1
            else:
                # If no constraints, all present count as OK for constraints
                constraint_ok_count = present_count

        type_rate = (type_ok_count / present_count) if present_count else 0.0
        constraint_rate = (constraint_ok_count / present_count) if present_count else 0.0

        # Field composite score per row (presence * weight + type * weight + constraint * weight)
        # We aggregate by using the rates directly
        field_score = (
            weights.get("required_presence", 0.5) * presence_rate +
            weights.get("type_correctness", 0.25) * type_rate +
            weights.get("constraint_compliance", 0.25) * constraint_rate
        )
        field_scores.append(field_score)

        metrics["fields"][field] = {
            "expected_type": ftype,
            "presence_rate": round(presence_rate * 100.0, 2),
            "type_correctness_rate": round(type_rate * 100.0, 2),
            "constraint_compliance_rate": round(constraint_rate * 100.0, 2),
            "field_score_pct": round(field_score * 100.0, 2),
        }

    # Average across fields to get overall score
    overall_score = (sum(field_scores) / len(field_scores)) if field_scores else 0.0
    metrics["overall_score_pct"] = round(overall_score * 100.0, 2)

    return metrics


# Main Calss

def parse_args() -> None:
    parser = argparse.ArgumentParser(description="Validate H3 res=8 via S3 Select vs 8-only file and schema.")
    parser.add_argument("--bucket-8-10", type=str, help="S3 bucket for the 8–10 GeoJSON.")
    parser.add_argument("--key-8-10", type=str, help="S3 key for the 8–10 GeoJSON.")
    parser.add_argument("--s3-validation", action="store_true", help="Use S3 for validation source.")
    parser.add_argument("--no-s3-validation", action="store_true", help="Use local file for validation.")
    parser.add_argument("--validation-use-select", action="store_true", help="Use S3 Select for the validation file.")
    parser.add_argument("--bucket-8", type=str, help="S3 bucket for the 8-only validation GeoJSON.")
    parser.add_argument("--key-8", type=str, help="S3 key for the 8-only validation GeoJSON.")
    parser.add_argument("--local-8-path", type=str, help="Local path for validation GeoJSON if not using S3.")
    parser.add_argument("--schema-path", type=str, help="Path to hex_schema.json.")
    parser.add_argument("--log-file", type=str, help="Log file path.")
    parser.add_argument("--output-csv", type=str, help="Where to write the res=8 CSV output.")
    parser.add_argument("--schema-metrics-json", type=str, help="Where to write schema metrics JSON.")
    parser.add_argument("--missing-h3-file", type=str, help="Where to write missing h3index list.")
    parser.add_argument("--extra-h3-file", type=str, help="Where to write extra h3index list.")
    parser.add_argument("--content-threshold", type=float, help="Coverage threshold in percent (default 99.0).")
    parser.add_argument("--aws-region", type=str, help="AWS region name (overrides environment).")
    parser.add_argument("--aws-endpoint-url", type=str, help="Custom S3 endpoint URL.")
    parser.add_argument("--compression", type=str, choices=["auto", "none", "gzip"], help="Compression handling for S3 Select input.")
    parser.add_argument("--sample-limit", type=int, help="Limit number of rows returned by S3 Select (dev/testing).")
    args = parser.parse_args()

    if args.bucket_8_10: CFG.bucket_8_10 = args.bucket_8_10
    if args.key_8_10: CFG.key_8_10 = args.key_8_10
    if args.s3_validation: CFG.s3_validation = True
    if args.no_s3_validation: CFG.s3_validation = False
    if args.validation_use_select: CFG.validation_use_select = True
    if args.bucket_8: CFG.bucket_8 = args.bucket_8
    if args.key_8: CFG.key_8 = args.key_8
    if args.local_8_path: CFG.local_8_path = args.local_8_path
    if args.schema_path: CFG.schema_path = args.schema_path
    if args.log_file: CFG.log_file = args.log_file
    if args.output_csv: CFG.output_csv = args.output_csv
    if args.schema_metrics_json: CFG.schema_metrics_json = args.schema_metrics_json
    if args.missing_h3_file: CFG.missing_h3_file = args.missing_h3_file
    if args.extra_h3_file: CFG.extra_h3_file = args.extra_h3_file
    if args.content_threshold is not None: CFG.content_threshold_pct = float(args.content_threshold)
    if args.aws_region: CFG.aws_region = args.aws_region
    if args.aws_endpoint_url: CFG.aws_endpoint_url = args.aws_endpoint_url
    if args.compression: CFG.compression = args.compression
    if args.sample_limit is not None: CFG.sample_limit = int(args.sample_limit)


def main() -> int:
    parse_args()
    setup_logging()
    logging.info("=== S3 Select Hex Res=8 Validation Script (Improved) ===")
    logging.info("Config: %s", json.dumps(asdict(CFG), indent=2, default=str))
    t0 = time.time()

    try:
        with timed_step("S3 Select resolution=8 features from 8-10 GeoJSON"):
            res8_df = s3_select_res8_hexes()

        with timed_step("Load validation 8-only GeoJSON"):
            val_df = load_validation_hexes()

        with timed_step("Content validation vs 8-only file"):
            coverage_pct, missing_ct, total_val, extras_ct = content_validation(res8_df, val_df)

        if coverage_pct < CFG.content_threshold_pct:
            logging.error(
                "Content coverage %.2f%% is below threshold %.2f%% (missing %d of %d; extras=%d).",
                coverage_pct, CFG.content_threshold_pct, missing_ct, total_val, extras_ct
            )
            return 1

        with timed_step("Schema validation & conformance scoring"):
            schema = load_schema()
            metrics = schema_conformance_metrics(res8_df, schema)
            score = metrics.get("overall_score_pct", 0.0)
            threshold = float(metrics.get("threshold_pct", 95.0))
            logging.info("Schema conformance score: %.2f%% (threshold=%.2f%%)", score, threshold)

            # Persist metrics
            with open(CFG.schema_metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logging.info("Wrote schema metrics to %s", CFG.schema_metrics_json)

            if score < threshold:
                logging.error(
                    "Schema conformance score %.2f%% is below threshold %.2f%%.",
                    score, threshold
                )
                return 1

        with timed_step("Save S3 Select res=8 result to CSV"):
            # Persist the normalized, deduped selection
            res8_df.to_csv(CFG.output_csv, index=False)
            logging.info("Wrote CSV to %s", CFG.output_csv)

        t1 = time.time()
        logging.info("Total wall-clock time: %.2f seconds", t1 - t0)
        logging.info("Script completed successfully.")
        return 0

    except Exception as e:
        logging.exception("Script failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
