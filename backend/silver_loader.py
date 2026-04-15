"""
Read Hive-partitioned trip silver Parquet under final_results/ and map to dashboard + ML frames.
Env (optional): REALTIME_FINAL_RESULTS_ROOT, REALTIME_SILVER_SUBDIR, SILVER_MAX_FILES, SILVER_ROW_CAP,
SILVER_PARQUET_BATCH_SIZE (rows per in-memory chunk; smaller uses less RAM),
REALTIME_USE_FINAL_RESULTS (0|1|unset=auto when parquet exists).
Dashboard silver path uses batched PyArrow reads (aggregate_silver_batched), not a full concat of all trips.
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULTS_ROOT = Path(
    os.environ.get(
        "REALTIME_FINAL_RESULTS_ROOT",
        os.environ.get("FINAL_RESULTS_ROOT", str(REPO_ROOT / "final_results")),
    )
)
SILVER_SUBDIR = os.environ.get(
    "REALTIME_SILVER_SUBDIR",
    os.environ.get("SILVER_SUBDIR", "trip_details_silver"),
)
SILVER_MAX_FILES = int(os.environ.get("SILVER_MAX_FILES", "50"))
SILVER_ROW_CAP = int(os.environ.get("SILVER_ROW_CAP", "500000"))
# Rows per PyArrow batch (lower = less RAM per step, slower)
SILVER_PARQUET_BATCH_SIZE = int(os.environ.get("SILVER_PARQUET_BATCH_SIZE", "65536"))


def silver_table_path() -> Path:
    return FINAL_RESULTS_ROOT / SILVER_SUBDIR


def silver_data_available() -> bool:
    root = silver_table_path()
    if not root.is_dir():
        return False
    return any(root.rglob("*.parquet"))


def use_final_results_silver() -> bool:
    v = os.environ.get("REALTIME_USE_FINAL_RESULTS")
    if v is not None and v.strip().lower() in ("0", "false", "no", "off"):
        return False
    if v is not None and v.strip().lower() in ("1", "true", "yes", "on"):
        return silver_data_available()
    return silver_data_available()


def _resolve_df_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _silver_file_sort_key(path: Path) -> tuple:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    s = str(path).replace("\\", "/").lower()
    y = mo = 0
    ym = re.search(r"year=(\d+)", s)
    mm = re.search(r"month=(\d+)", s)
    if ym:
        y = int(ym.group(1))
    if mm:
        mo = int(mm.group(1))
    return (-y, -mo, -mtime)


def list_silver_parquet_files() -> List[Path]:
    """Partition Parquet paths under the silver table (newest partitions first, capped)."""
    root = silver_table_path()
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*.parquet") if p.is_file() and "_SUCCESS" not in p.name]
    if not files:
        return []
    files = sorted(files, key=_silver_file_sort_key)
    return files[: max(1, SILVER_MAX_FILES)]


def _pick_schema_column(names: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        if c in names:
            return c
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _day_split_delta_pct(
    hourly_cnt: Dict[int, int],
    hourly_rev: Dict[int, float],
    revenue: bool,
) -> float:
    """% difference afternoon (12–23h) vs morning (0–11h) in the loaded sample."""
    if not hourly_cnt:
        return 0.0
    if revenue:
        a = sum(hourly_rev.get(h, 0.0) for h in range(0, 12))
        b = sum(hourly_rev.get(h, 0.0) for h in range(12, 24))
    else:
        a = sum(hourly_cnt.get(h, 0) for h in range(0, 12))
        b = sum(hourly_cnt.get(h, 0) for h in range(12, 24))
    denom = max(a, 1e-9)
    return round(100.0 * (b - a) / denom, 1)


def _kpis_from_totals(
    n: int,
    revenue: float,
    dur_sum: float,
    dur_n: int,
    hourly_cnt: Optional[Dict[int, int]] = None,
    hourly_rev: Optional[Dict[int, float]] = None,
    ingest_wall_ms: float = 0.0,
    parquet_bytes_mb: float = 0.0,
) -> Dict[str, Any]:
    if n <= 0:
        return {
            "total_trips": 0,
            "total_trips_change": 0.0,
            "total_revenue": 0.0,
            "total_revenue_change": 0.0,
            "avg_trip_duration": 0.0,
            "avg_fare": 0.0,
            "spark_processing_time_ms": 0,
            "data_mart_size_mb": 0.0,
            "source": "final_results_silver_empty",
        }
    hc = hourly_cnt or {}
    hr = hourly_rev or {}
    avg_fare = revenue / n
    avg_dur = 0.0
    if dur_n > 0:
        avg_dur = round(float(dur_sum / dur_n), 1)
    trips_delta = _day_split_delta_pct(hc, hr, revenue=False)
    rev_delta = _day_split_delta_pct(hc, hr, revenue=True)
    return {
        "total_trips": n,
        "total_trips_change": trips_delta,
        "total_revenue": round(revenue, 2),
        "total_revenue_change": rev_delta,
        "avg_trip_duration": avg_dur,
        "avg_fare": round(avg_fare, 2),
        "spark_processing_time_ms": int(round(max(0.0, ingest_wall_ms))),
        "data_mart_size_mb": round(max(0.0, parquet_bytes_mb), 2),
        "source": "final_results_silver",
    }


def _hourly_list_from_maps(
    hourly_cnt: Dict[int, int],
    hourly_rev: Dict[int, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in sorted(hourly_cnt.keys()):
        out.append(
            {
                "hour": int(h),
                "hour_label": f"{int(h):02d}:00",
                "trip_count": int(hourly_cnt[h]),
                "revenue": round(float(hourly_rev.get(h, 0.0)), 2),
            }
        )
    return out


def _zones_list_from_maps(
    zone_cnt: Dict[int, int],
    zone_rev: Dict[int, float],
    zone_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for zid, tc in zone_cnt.items():
        rev = zone_rev.get(zid, 0.0)
        avg_f = rev / max(tc, 1)
        data.append(
            {
                "zone_id": int(zid),
                "zone_name": zone_names.get(int(zid), f"Zone {zid}"),
                "trip_count": int(tc),
                "avg_fare": round(float(avg_f), 2),
                "density": round(int(tc) / 100.0, 2),
            }
        )
    return sorted(data, key=lambda x: x["trip_count"], reverse=True)


def _ml_frame_from_bucket_map(
    ml_buckets: Dict[Tuple[int, pd.Timestamp], List[float]],
) -> pd.DataFrame:
    from model_registry import FEATURE_COLS

    if not ml_buckets:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (zid, bucket), (cnt, rev) in sorted(
        ml_buckets.items(),
        key=lambda kv: (kv[0][0], kv[0][1]),
    ):
        bt = pd.Timestamp(bucket)
        rows.append(
            {
                "zone_id": int(zid),
                "bucket_time": bt,
                "demand_count": int(cnt),
                "total_revenue": float(rev),
                "pickup_hour": int(bt.hour),
                "dayofweek": int(bt.dayofweek),
                "is_weekend": int(1 if bt.dayofweek >= 5 else 0),
            }
        )
    g = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in g.columns:
            return pd.DataFrame()
    zcol = os.environ.get("REALTIME_ZONE_COLUMN", "zone_id")
    if zcol != "zone_id":
        g = g.rename(columns={"zone_id": zcol})
    return g[[zcol, "bucket_time"] + FEATURE_COLS].copy()


def aggregate_silver_batched(
    zone_names: Dict[int, str],
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]]:
    """
    Stream Parquet files in batches: only small chunks in RAM; merge into counters / hourly ML buckets.
    Returns (kpis, hourly_list, zones_list, df_ml, silver_ingest) or None if no data.
    """
    files = list_silver_parquet_files()
    if not files or pq is None:
        return None

    ingest_t0 = time.perf_counter()
    parquet_bytes = 0.0
    for fp in files:
        try:
            parquet_bytes += float(fp.stat().st_size)
        except OSError:
            continue
    parquet_bytes_mb = parquet_bytes / (1024.0 * 1024.0)

    cap = max(1, SILVER_ROW_CAP)
    batch_size = max(1024, SILVER_PARQUET_BATCH_SIZE)

    hourly_cnt: DefaultDict[int, int] = defaultdict(int)
    hourly_rev: DefaultDict[int, float] = defaultdict(float)
    zone_cnt: DefaultDict[int, int] = defaultdict(int)
    zone_rev: DefaultDict[int, float] = defaultdict(float)
    ml_buckets: DefaultDict[Tuple[int, pd.Timestamp], List[float]] = defaultdict(lambda: [0, 0.0])

    total_rows = 0
    revenue_sum = 0.0
    dur_sum = 0.0
    dur_n = 0
    files_read_ok = 0
    batches = 0

    for fp in files:
        if total_rows >= cap:
            break
        try:
            schema = pq.read_schema(fp)
        except Exception as e:
            print(f"silver_loader: schema skip {fp}: {e}")
            continue
        names = list(schema.names)
        ts_col = _pick_schema_column(
            names,
            ["pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"],
        )
        zone_col = _pick_schema_column(
            names,
            ["PULocationID", "pulocationid", "pickup_location_id", "PickupLocationID"],
        )
        fare_col = _pick_schema_column(names, ["fare_amount", "total_amount", "Fare_amount", "Total_amount"])
        if not ts_col or not zone_col or not fare_col:
            continue
        read_cols = [ts_col, zone_col, fare_col]
        dur_col = _pick_schema_column(names, ["trip_duration_mins", "trip_duration"])
        if dur_col:
            read_cols.append(dur_col)

        try:
            pf = pq.ParquetFile(fp)
        except Exception as e:
            print(f"silver_loader: open skip {fp}: {e}")
            continue

        file_ok = False
        try:
            for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
                if total_rows >= cap:
                    break
                chunk = batch.to_pandas().reset_index(drop=True)
                if chunk.empty:
                    continue
                batches += 1
                pts = pd.to_datetime(chunk[ts_col], errors="coerce")
                zids = pd.to_numeric(chunk[zone_col], errors="coerce")
                fares = pd.to_numeric(chunk[fare_col], errors="coerce")
                norm = pd.DataFrame({"pickup_ts": pts, "zone_id": zids, "fare": fares})
                if dur_col and dur_col in chunk.columns:
                    norm["_dur"] = pd.to_numeric(chunk[dur_col], errors="coerce")
                norm = norm.dropna(subset=["pickup_ts", "zone_id", "fare"])
                if norm.empty:
                    del chunk, norm
                    continue
                norm["zone_id"] = norm["zone_id"].astype(int)
                take = min(len(norm), cap - total_rows)
                if take < len(norm):
                    norm = norm.iloc[:take].copy()

                total_rows += len(norm)
                revenue_sum += float(norm["fare"].sum())

                if "_dur" in norm.columns:
                    valid = norm["_dur"].dropna()
                    if len(valid):
                        dur_sum += float(valid.sum())
                        dur_n += int(valid.count())
                    norm = norm.drop(columns=["_dur"])

                gh = norm.groupby(norm["pickup_ts"].dt.hour, sort=False)["fare"].agg(["size", "sum"])
                for hour, row in gh.iterrows():
                    hi = int(hour)
                    hourly_cnt[hi] += int(row["size"])
                    hourly_rev[hi] += float(row["sum"])

                gz = norm.groupby("zone_id", sort=False)["fare"].agg(["size", "sum"])
                for zid, row in gz.iterrows():
                    zi = int(zid)
                    zone_cnt[zi] += int(row["size"])
                    zone_rev[zi] += float(row["sum"])

                norm_bt = norm.assign(bucket=norm["pickup_ts"].dt.floor("h"))
                gm = norm_bt.groupby(["zone_id", "bucket"], sort=False)["fare"].agg(["size", "sum"])
                for (zid, bucket), row in gm.iterrows():
                    k = (int(zid), pd.Timestamp(bucket))
                    ml_buckets[k][0] += int(row["size"])
                    ml_buckets[k][1] += float(row["sum"])

                del chunk, norm, gh, gz, gm, norm_bt
                file_ok = True
        except Exception as e:
            print(f"silver_loader: batch read failed {fp}, trying pandas columns=… : {e}")
            try:
                sub = pd.read_parquet(fp, columns=read_cols).reset_index(drop=True)
                sub = sub.iloc[: max(0, cap - total_rows)]
                if sub.empty:
                    continue
                norm = pd.DataFrame(
                    {
                        "pickup_ts": pd.to_datetime(sub[ts_col], errors="coerce"),
                        "zone_id": pd.to_numeric(sub[zone_col], errors="coerce"),
                        "fare": pd.to_numeric(sub[fare_col], errors="coerce"),
                    }
                )
                if dur_col and dur_col in sub.columns:
                    norm["_dur"] = pd.to_numeric(sub[dur_col], errors="coerce")
                norm = norm.dropna(subset=["pickup_ts", "zone_id", "fare"])
                norm["zone_id"] = norm["zone_id"].astype(int)
                if norm.empty:
                    continue
                total_rows += len(norm)
                revenue_sum += float(norm["fare"].sum())
                if "_dur" in norm.columns:
                    d = norm["_dur"].dropna()
                    if len(d):
                        dur_sum += float(d.sum())
                        dur_n += int(d.count())
                    norm = norm.drop(columns=["_dur"])
                gh = norm.groupby(norm["pickup_ts"].dt.hour, sort=False)["fare"].agg(["size", "sum"])
                for hour, row in gh.iterrows():
                    hi = int(hour)
                    hourly_cnt[hi] += int(row["size"])
                    hourly_rev[hi] += float(row["sum"])
                gz = norm.groupby("zone_id", sort=False)["fare"].agg(["size", "sum"])
                for zid, row in gz.iterrows():
                    zi = int(zid)
                    zone_cnt[zi] += int(row["size"])
                    zone_rev[zi] += float(row["sum"])
                norm_bt = norm.assign(bucket=norm["pickup_ts"].dt.floor("h"))
                gm = norm_bt.groupby(["zone_id", "bucket"], sort=False)["fare"].agg(["size", "sum"])
                for (zid, bucket), row in gm.iterrows():
                    k = (int(zid), pd.Timestamp(bucket))
                    ml_buckets[k][0] += int(row["size"])
                    ml_buckets[k][1] += float(row["sum"])
                file_ok = True
                del sub, norm
            except Exception as e2:
                print(f"silver_loader: pandas fallback skip {fp}: {e2}")
                continue

        if file_ok:
            files_read_ok += 1

    if total_rows <= 0:
        return None

    ingest_wall_ms = (time.perf_counter() - ingest_t0) * 1000.0
    hc = dict(hourly_cnt)
    hr = dict(hourly_rev)
    kpis = _kpis_from_totals(
        total_rows,
        revenue_sum,
        dur_sum,
        dur_n,
        hourly_cnt=hc,
        hourly_rev=hr,
        ingest_wall_ms=ingest_wall_ms,
        parquet_bytes_mb=parquet_bytes_mb,
    )
    hourly_list = _hourly_list_from_maps(hc, hr)
    zones_list = _zones_list_from_maps(dict(zone_cnt), dict(zone_rev), zone_names)
    df_ml = _ml_frame_from_bucket_map(dict(ml_buckets))

    rps = (total_rows / (ingest_wall_ms / 1000.0)) if ingest_wall_ms > 1.0 else float(total_rows)

    silver_ingest: Dict[str, Any] = {
        "ingest_mode": "batched_pyarrow",
        "partition_files_selected": len(files),
        "files_read_ok": files_read_ok,
        "sample_paths": [str(p) for p in files[:8]],
        "rows_processed": int(total_rows),
        "parquet_batch_size": batch_size,
        "batches_processed": batches,
        "ml_bucket_count": len(ml_buckets),
        "silver_row_cap": cap,
        "ingest_wall_ms": round(ingest_wall_ms, 2),
        "parquet_candidate_bytes_mb": round(parquet_bytes_mb, 3),
        "ingest_rows_per_sec": round(rps, 1),
    }
    return kpis, hourly_list, zones_list, df_ml, silver_ingest


def read_silver_trips_df() -> pd.DataFrame:
    """
    Full-trip frame (high RAM). Prefer aggregate_silver_batched() for the dashboard.
    Reads one partition at a time and stops at SILVER_ROW_CAP — still bounded but heavier than batched aggregates.
    """
    files = list_silver_parquet_files()
    if not files:
        return pd.DataFrame()
    cap = max(1, SILVER_ROW_CAP)
    parts: List[pd.DataFrame] = []
    rows = 0
    for fp in files:
        if rows >= cap:
            break
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"silver_loader: skip {fp}: {e}")
            continue
        if df.empty:
            continue
        need = min(len(df), cap - rows)
        if need < len(df):
            df = df.iloc[:need].copy()
        parts.append(df)
        rows += len(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def normalize_silver_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts_col = _resolve_df_column(
        df,
        ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"],
    )
    zone_col = _resolve_df_column(
        df,
        ["PULocationID", "pulocationid", "pickup_location_id", "PickupLocationID"],
    )
    fare_col = _resolve_df_column(df, ["fare_amount", "total_amount", "Fare_amount", "Total_amount"])
    if not ts_col or not zone_col or not fare_col:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "pickup_ts": pd.to_datetime(df[ts_col], errors="coerce"),
            "zone_id": pd.to_numeric(df[zone_col], errors="coerce"),
            "fare": pd.to_numeric(df[fare_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["pickup_ts", "zone_id", "fare"])
    out["zone_id"] = out["zone_id"].astype(int)
    return out.reset_index(drop=True)


def silver_to_kpis(norm: pd.DataFrame) -> Dict[str, Any]:
    if norm.empty:
        return {
            "total_trips": 0,
            "total_trips_change": 0.0,
            "total_revenue": 0.0,
            "total_revenue_change": 0.0,
            "avg_trip_duration": 0.0,
            "avg_fare": 0.0,
            "spark_processing_time_ms": 0,
            "data_mart_size_mb": 0.0,
            "source": "final_results_silver_empty",
        }
    n = int(len(norm))
    revenue = float(norm["fare"].sum())
    dur_sum = 0.0
    dur_n = 0
    if "trip_duration_mins" in norm.columns:
        d = pd.to_numeric(norm["trip_duration_mins"], errors="coerce").dropna()
        dur_n = int(d.count())
        if dur_n:
            dur_sum = float(d.sum())
    gh = norm.groupby(norm["pickup_ts"].dt.hour, sort=False)["fare"].agg(["size", "sum"])
    hourly_cnt = {int(h): int(r["size"]) for h, r in gh.iterrows()}
    hourly_rev = {int(h): float(r["sum"]) for h, r in gh.iterrows()}
    return _kpis_from_totals(
        n,
        revenue,
        dur_sum,
        dur_n,
        hourly_cnt=hourly_cnt,
        hourly_rev=hourly_rev,
        ingest_wall_ms=0.0,
        parquet_bytes_mb=0.0,
    )


def silver_hourly_trends(norm: pd.DataFrame) -> List[Dict[str, Any]]:
    if norm.empty:
        return []
    tmp = norm.assign(pickup_hour=norm["pickup_ts"].dt.hour)
    g = tmp.groupby("pickup_hour", as_index=False).agg(
        trip_count=("fare", "size"),
        revenue=("fare", "sum"),
    )
    out: List[Dict[str, Any]] = []
    for _, row in g.iterrows():
        h = int(row["pickup_hour"])
        out.append(
            {
                "hour": h,
                "hour_label": f"{h:02d}:00",
                "trip_count": int(row["trip_count"]),
                "revenue": round(float(row["revenue"]), 2),
            }
        )
    return sorted(out, key=lambda x: x["hour"])


def silver_zone_heatmap(norm: pd.DataFrame, zone_names: Dict[int, str]) -> List[Dict[str, Any]]:
    if norm.empty:
        return []
    g = norm.groupby("zone_id", as_index=False).agg(
        trip_count=("fare", "size"),
        avg_fare=("fare", "mean"),
    )
    data: List[Dict[str, Any]] = []
    for _, row in g.iterrows():
        zid = int(row["zone_id"])
        tc = int(row["trip_count"])
        data.append(
            {
                "zone_id": zid,
                "zone_name": zone_names.get(zid, f"Zone {zid}"),
                "trip_count": tc,
                "avg_fare": round(float(row["avg_fare"]), 2),
                "density": round(tc / 100.0, 2),
            }
        )
    return sorted(data, key=lambda x: x["trip_count"], reverse=True)


def derive_ml_timeseries(norm: pd.DataFrame) -> pd.DataFrame:
    """Hourly buckets per zone: zone_id + FEATURE_COLS for model_registry."""
    from model_registry import FEATURE_COLS

    if norm.empty:
        return pd.DataFrame()
    tmp = norm.assign(bucket_time=norm["pickup_ts"].dt.floor("h"))
    g = tmp.groupby(["zone_id", "bucket_time"], as_index=False).agg(
        demand_count=("fare", "count"),
        total_revenue=("fare", "sum"),
    )
    g["pickup_hour"] = g["bucket_time"].dt.hour.astype(int)
    g["dayofweek"] = g["bucket_time"].dt.dayofweek.astype(int)
    g["is_weekend"] = (g["dayofweek"] >= 5).astype(int)
    g = g.sort_values(["zone_id", "bucket_time"]).reset_index(drop=True)
    for c in FEATURE_COLS:
        if c not in g.columns:
            return pd.DataFrame()
    zcol = os.environ.get("REALTIME_ZONE_COLUMN", "zone_id")
    if zcol != "zone_id":
        g = g.rename(columns={"zone_id": zcol})
    return g[[zcol, "bucket_time"] + FEATURE_COLS].copy()
