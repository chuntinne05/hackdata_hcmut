import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from model_registry import (
    FEATURE_COLS,
    NestedModelRegistry,
    TIME_STEPS,
    apply_feature_scaling,
    build_scaled_window,
)
from silver_loader import (
    SILVER_ROW_CAP,
    aggregate_silver_batched,
    silver_data_available,
    silver_table_path,
    use_final_results_silver,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ZONE_CSV_COL = os.environ.get("REALTIME_ZONE_COLUMN", "zone_id")
WS_TICK_SECONDS = float(os.environ.get("WS_TICK_SECONDS", "2.0"))
DEFAULT_BUNDLE_ID = os.environ.get("REALTIME_DEFAULT_BUNDLE", "100")

registry = NestedModelRegistry(MODELS_DIR)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE_HINTS: Dict[str, str] = {
    "100": "Ưu tiên zone 100 (Garment District) nếu train theo zone.",
    "230": "Ưu tiên zone 230 (Times Sq / Theatre District).",
    "237": "Ưu tiên zone 237 (Upper East Side South).",
}


def _bundle_catalog(bundle_ids: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for bid in bundle_ids:
        out.append(
            {
                "id": bid,
                "hint": BUNDLE_HINTS.get(bid, f"Bundle `{bid}` — kiểm tra thư mục backend/models/{bid}/."),
            }
        )
    return out


def _ml_sample_time_range(df_ml: pd.DataFrame) -> Optional[Dict[str, str]]:
    if df_ml.empty or "bucket_time" not in df_ml.columns:
        return None
    s = pd.to_datetime(df_ml["bucket_time"], errors="coerce").dropna()
    if s.empty:
        return None
    return {"min": s.min().strftime("%Y-%m-%d %H:%M"), "max": s.max().strftime("%Y-%m-%d %H:%M")}


def _single_series_metrics(yt: np.ndarray, yp: np.ndarray) -> Optional[Dict[str, float]]:
    if yt.size < 2 or yp.size != yt.size:
        return None
    err = yt.astype(np.float64) - yp.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ymean = float(np.mean(yt))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt.astype(np.float64) - ymean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2_score": round(r2, 4)}


def _per_model_metrics_from_validation(points: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if len(points) < 2:
        return {}
    yt = np.array([float(p["actual"]) for p in points], dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for label, pred_key in (
        ("ensemble", "predicted"),
        ("lstm", "pred_lstm"),
        ("xgboost", "pred_xgb"),
        ("random_forest", "pred_rf"),
    ):
        if pred_key not in points[0]:
            continue
        yp = np.array([float(p[pred_key]) for p in points], dtype=np.float64)
        m = _single_series_metrics(yt, yp)
        if m:
            out[label] = m
    return out


TAXI_ZONES = {
    4: "Alphabet City",
    13: "Battery Park",
    24: "Bloomingdale",
    43: "Central Park",
    48: "Clinton East",
    68: "East Chelsea",
    79: "East Village",
    87: "Financial District South",
    88: "Financial District North",
    100: "Garment District",
    107: "Gramercy",
    113: "Greenwich Village North",
    125: "Harlem North",
    137: "Kips Bay",
    140: "Lenox Hill East",
    142: "Lincoln Square East",
    143: "Lincoln Square West",
    148: "Lower East Side",
    151: "Manhattanville",
    158: "Meatpacking/West Village West",
    161: "Midtown Center",
    162: "Midtown East",
    163: "Midtown North",
    164: "Midtown South",
    166: "Morningside Heights",
    170: "Murray Hill",
    186: "Penn Station/Madison Sq West",
    202: "Seaport",
    209: "SoHo",
    211: "South Williamsburg",
    224: "Stuy Town/Peter Cooper Village",
    229: "Sutton Place/Turtle Bay North",
    230: "Times Sq/Theatre District",
    231: "TriBeCa/Civic Center",
    232: "Two Bridges/Seward Park",
    233: "UN/Turtle Bay South",
    234: "Union Sq",
    236: "Upper East Side North",
    237: "Upper East Side South",
    238: "Upper West Side North",
    239: "Upper West Side South",
    243: "Washington Heights North",
    244: "Washington Heights South",
    246: "West Chelsea/Hudson Yards",
    249: "West Village",
}


def _fit_metrics_from_validation_points(
    points: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, float], float]]:
    """RMSE / MAE / R² and a bounded confidence score from walk-forward validation points."""
    if len(points) < 2:
        return None
    yt = np.array([float(p["actual"]) for p in points], dtype=np.float64)
    yp = np.array([float(p["predicted"]) for p in points], dtype=np.float64)
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ymean = float(np.mean(yt))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - ymean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    denom = max(float(np.mean(np.abs(yt))), 1.0)
    nrmse = rmse / denom
    confidence = max(0.0, min(0.99, 1.0 / (1.0 + nrmse)))
    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
    }
    return metrics, round(confidence, 3)


def _enrich_prediction_metrics(
    pred: Optional[Dict[str, Any]],
    validation_points: List[Dict[str, Any]],
) -> None:
    if pred is None or pred.get("error") or "predictions" not in pred:
        return
    pred["executor_nodes"] = 1
    fit = _fit_metrics_from_validation_points(validation_points)
    if fit:
        pred["model_metrics"], pred["confidence_score"] = fit
    else:
        pred["model_metrics"] = {"rmse": 0.0, "mae": 0.0, "r2_score": 0.0}
        pred["confidence_score"] = 0.0
    pmm = _per_model_metrics_from_validation(validation_points)
    if pmm:
        pred["validation_metrics_by_model"] = pmm


def _streaming_from_silver(silver_ingest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Honest ingest stats (Parquet batch read), not Kafka."""
    if not silver_ingest:
        return {
            "kafka_topics": [],
            "messages_per_second": 0,
            "processing_latency_ms": 0,
            "batch_interval_seconds": int(WS_TICK_SECONDS),
            "offset_lag": 0,
            "status": "NO_SILVER_DATA",
            "last_checkpoint": datetime.now().isoformat(),
        }
    topics = [Path(p).name for p in silver_ingest.get("sample_paths", [])[:6]]
    if not topics:
        topics = ["(no sample paths)"]
    sel = int(silver_ingest.get("partition_files_selected", 0))
    ok = int(silver_ingest.get("files_read_ok", 0))
    return {
        "kafka_topics": topics,
        "messages_per_second": int(round(float(silver_ingest.get("ingest_rows_per_sec", 0)))),
        "processing_latency_ms": int(round(float(silver_ingest.get("ingest_wall_ms", 0)))),
        "batch_interval_seconds": int(WS_TICK_SECONDS),
        "offset_lag": max(0, sel - ok),
        "status": "PARQUET_INGEST",
        "last_checkpoint": datetime.now().isoformat(),
    }


def _cluster_from_silver(
    silver_ingest: Optional[Dict[str, Any]],
    df_ml: pd.DataFrame,
) -> Dict[str, Any]:
    if not silver_ingest:
        return {
            "active_workers": 0,
            "total_cores": 0,
            "total_memory_gb": 0.0,
            "memory_used_gb": 0.0,
            "cpu_utilization_pct": 0.0,
            "active_jobs": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "data_processed_gb": 0.0,
            "avg_task_duration_ms": 0,
            "network_throughput_mbps": 0.0,
            "silver_rows": 0,
            "silver_ml_buckets": 0,
            "parquet_size_mb": 0.0,
        }
    rows = int(silver_ingest.get("rows_processed", 0))
    buckets = int(silver_ingest.get("ml_bucket_count", 0))
    batches = int(silver_ingest.get("batches_processed", 0))
    sel = int(silver_ingest.get("partition_files_selected", 0))
    ok = int(silver_ingest.get("files_read_ok", 0))
    pmb = float(silver_ingest.get("parquet_candidate_bytes_mb", 0.0))
    ingest_ms = float(silver_ingest.get("ingest_wall_ms", 0.0))
    pct_ok = round(100.0 * ok / max(sel, 1), 1)
    return {
        "active_workers": ok,
        "total_cores": batches,
        "total_memory_gb": round(pmb, 2),
        "memory_used_gb": round(pmb, 2),
        "cpu_utilization_pct": pct_ok,
        "active_jobs": sel,
        "completed_tasks": rows,
        "failed_tasks": max(0, sel - ok),
        "data_processed_gb": round(pmb / 1024.0, 4),
        "avg_task_duration_ms": int(round(ingest_ms / max(batches, 1))),
        "network_throughput_mbps": 0.0,
        "silver_rows": rows,
        "silver_ml_buckets": buckets,
        "parquet_size_mb": round(pmb, 2),
    }


def _kpis_when_no_silver() -> Dict[str, Any]:
    return {
        "total_trips": 0,
        "total_trips_change": 0.0,
        "total_revenue": 0.0,
        "total_revenue_change": 0.0,
        "avg_trip_duration": 0.0,
        "avg_fare": 0.0,
        "spark_processing_time_ms": 0,
        "data_mart_size_mb": 0.0,
        "source": "final_results_unavailable",
    }


def _nonneg_demand_round(x: float) -> int:
    return int(max(0.0, round(float(x))))


def _revenue_rate_from_tail(df: pd.DataFrame, zone_id: int) -> float:
    sub = df[df[ZONE_CSV_COL] == zone_id].copy()
    if sub.empty or "total_revenue" not in sub.columns or "demand_count" not in sub.columns:
        return 0.0
    if "bucket_time" in sub.columns:
        sub = sub.sort_values("bucket_time", kind="mergesort")
    tail = sub.tail(TIME_STEPS)
    if tail.empty:
        return 0.0
    d = tail["demand_count"].astype(float).clip(lower=1.0)
    r = tail["total_revenue"].astype(float)
    return float((r / d).mean())


def run_inference(bundle_id: str, zone_id: int, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    bundle, err = registry.get(bundle_id)
    if bundle is None or df.empty:
        return None
    scaled, werr = build_scaled_window(df, ZONE_CSV_COL, zone_id, bundle.feature_scaler)
    if scaled is None:
        return {"error": werr or err, "bundle_id": bundle_id, "zone_id": zone_id}
    t0 = time.time()
    preds = bundle.predict_from_scaled_window(scaled)
    ms = (time.time() - t0) * 1000
    zone_name = TAXI_ZONES.get(zone_id, f"Zone {zone_id}")
    rate = _revenue_rate_from_tail(df, zone_id)
    rev_est = {
        "lstm": round(float(preds["lstm"]) * rate, 2),
        "xgboost": round(float(preds["xgboost"]) * rate, 2),
        "random_forest": round(float(preds["random_forest"]) * rate, 2),
        "ensemble": round(float(preds["ensemble"]) * rate, 2),
        "avg_revenue_per_trip_window": round(rate, 4),
    }
    return {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "bundle_id": bundle_id,
        "timestamp": datetime.now().isoformat(),
        "predictions": {
            "lstm": _nonneg_demand_round(preds["lstm"]),
            "xgboost": _nonneg_demand_round(preds["xgboost"]),
            "random_forest": _nonneg_demand_round(preds["random_forest"]),
            "ensemble": _nonneg_demand_round(preds["ensemble"]),
        },
        "revenue_estimates": rev_est,
        "inference_time_ms": round(ms, 2),
    }


def _validation_ts_label(row: pd.Series, fallback_index: int) -> str:
    if "bucket_time" in row.index and pd.notna(row["bucket_time"]):
        ts = pd.Timestamp(row["bucket_time"])
        return ts.strftime("%Y-%m-%d %H:%M")
    return f"step {fallback_index}"


def validation_from_csv(bundle_id: str, zone_id: int, df: pd.DataFrame, max_points: int = 25):
    bundle, _ = registry.get(bundle_id)
    if bundle is None or df.empty:
        return []
    sub = df[df[ZONE_CSV_COL] == zone_id].copy()
    if "bucket_time" in sub.columns:
        sub = sub.sort_values("bucket_time", kind="mergesort").reset_index(drop=True)
    else:
        sub = sub.reset_index(drop=True)
    if len(sub) < TIME_STEPS + 2:
        return []
    points = []
    start = max(TIME_STEPS, len(sub) - max_points)
    for i in range(start, len(sub)):
        window = sub.iloc[i - TIME_STEPS : i]
        raw = window[FEATURE_COLS].values.astype(np.float64)
        scaled = apply_feature_scaling(bundle.feature_scaler, raw)
        preds = bundle.predict_from_scaled_window(scaled)
        actual = float(sub.iloc[i]["demand_count"])
        row_i = sub.iloc[i]
        dwin = window["demand_count"].astype(float).clip(lower=1.0)
        rwin = window["total_revenue"].astype(float) if "total_revenue" in window.columns else pd.Series([0.0] * len(window))
        rate = float((rwin / dwin).mean()) if len(window) else 0.0
        pe = _nonneg_demand_round(preds["ensemble"])
        pl = _nonneg_demand_round(preds["lstm"])
        px = _nonneg_demand_round(preds["xgboost"])
        pr = _nonneg_demand_round(preds["random_forest"])
        act_rev = float(row_i["total_revenue"]) if "total_revenue" in row_i.index and pd.notna(row_i.get("total_revenue")) else 0.0
        points.append(
            {
                "timestamp": _validation_ts_label(row_i, i),
                "actual": int(round(actual)),
                "predicted": pe,
                "pred_lstm": pl,
                "pred_xgb": px,
                "pred_rf": pr,
                "actual_revenue": round(act_rev, 2),
                "predicted_revenue": round(float(pe) * rate, 2),
                "is_forecast": i == len(sub) - 1,
            }
        )
    return points[-max_points:]


def _static_source_meta() -> Dict[str, Any]:
    st = silver_table_path()
    return {
        "data_source_policy": "final_results_silver_only",
        "final_results_root": str(st.parent),
        "silver_subdir": st.name,
        "silver_table_dir": str(st),
        "silver_try": use_final_results_silver(),
        "silver_data_available": silver_data_available(),
        "realtime_use_final_results": os.environ.get("REALTIME_USE_FINAL_RESULTS", "auto"),
        "zone_column": ZONE_CSV_COL,
    }


def _tick_source_meta(
    data_source: str,
    df_ml: pd.DataFrame,
    silver_ingest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = {**_static_source_meta(), "data_source": data_source}
    meta["ml_frame_rows"] = int(len(df_ml)) if not df_ml.empty else 0
    meta["silver_row_cap"] = int(SILVER_ROW_CAP)
    meta["ml_sample_time_range"] = _ml_sample_time_range(df_ml)
    meta["silver_ingest"] = silver_ingest
    return meta


def _load_from_silver() -> Optional[tuple]:
    """Returns (kpis, hourly_list, zones_list, data_source, df_ml, silver_ingest) or None if unavailable."""
    if not use_final_results_silver():
        return None
    out = aggregate_silver_batched(TAXI_ZONES)
    if out is None:
        return None
    kpis, hourly_list, zones_list, df_ml, silver_ingest = out
    silver_ingest = {
        **silver_ingest,
        "ml_frame_rows": int(len(df_ml)) if not df_ml.empty else 0,
    }
    return (
        kpis,
        hourly_list,
        zones_list,
        kpis.get("source", "final_results_silver"),
        df_ml,
        silver_ingest,
    )


def get_ml_dataframe_for_inference() -> pd.DataFrame:
    """ML frame built only from Hive-style silver Parquet under final_results/."""
    pack = _load_from_silver()
    if pack is not None:
        _, _, _, _, df_ml, _ = pack
        if df_ml is not None and not df_ml.empty:
            return df_ml
    return pd.DataFrame()


def build_dashboard_payload(bundle_id: str, zone_id: int) -> Dict[str, Any]:
    """KPIs, charts, and ML inputs come only from final_results silver (see silver_loader)."""
    silver_pack = _load_from_silver()
    silver_ingest: Optional[Dict[str, Any]] = None
    if silver_pack is not None:
        kpis, hourly_list, zones_list, data_source, df_ml, silver_ingest = silver_pack
        hourly = {"data": list(hourly_list or [])}
        zones = {"data": list(zones_list or [])}
    else:
        kpis = _kpis_when_no_silver()
        hourly = {"data": []}
        zones = {"data": []}
        data_source = "final_results_unavailable"
        df_ml = pd.DataFrame()

    val = validation_from_csv(bundle_id, zone_id, df_ml) if not df_ml.empty else []
    pred = run_inference(bundle_id, zone_id, df_ml)
    _enrich_prediction_metrics(pred, val)

    source_meta = _tick_source_meta(data_source, df_ml, silver_ingest)
    bundle_catalog = _bundle_catalog(registry.list_bundles())

    return {
        "type": "tick",
        "ts": datetime.now().isoformat(),
        "bundle_id": bundle_id,
        "zone_id": zone_id,
        "data_source": data_source,
        "source_meta": source_meta,
        "bundle_catalog": bundle_catalog,
        "kpis": kpis,
        "hourlyTrends": hourly["data"],
        "zoneHeatmap": zones["data"],
        "prediction": pred,
        "validationHistory": val,
        "streamingStatus": _streaming_from_silver(silver_ingest),
        "clusterHealth": _cluster_from_silver(silver_ingest, df_ml),
        "registry": {"bundles": registry.list_bundles(), "loaded": registry.loaded_bundle_ids()},
    }


@app.get("/api/meta/bundles")
def list_model_bundles():
    return {"bundles": registry.list_bundles()}


@app.get("/api/overview/kpis")
def get_overview_kpis(bundle_id: str = DEFAULT_BUNDLE_ID, zone_id: int = 161):
    p = build_dashboard_payload(bundle_id, zone_id)
    return p["kpis"]


@app.get("/api/overview/hourly-trends")
def get_hourly_trends(bundle_id: str = DEFAULT_BUNDLE_ID, zone_id: int = 161):
    p = build_dashboard_payload(bundle_id, zone_id)
    return {"data": p["hourlyTrends"]}


@app.get("/api/overview/zone-heatmap")
def get_zone_heatmap(bundle_id: str = DEFAULT_BUNDLE_ID, zone_id: int = 161):
    p = build_dashboard_payload(bundle_id, zone_id)
    return {"data": p["zoneHeatmap"]}


@app.get("/api/forecast/predict")
def run_forecast_prediction(zone_id: int = 161, hours_ahead: int = 1, bundle_id: str = DEFAULT_BUNDLE_ID):
    df = get_ml_dataframe_for_inference()
    pred = run_inference(bundle_id, zone_id, df)
    if pred is None:
        return {
            "zone_id": zone_id,
            "bundle_id": bundle_id,
            "hours_ahead": hours_ahead,
            "error": "No model or insufficient final_results silver data for this zone",
        }
    if "error" in pred:
        pred["hours_ahead"] = hours_ahead
        return pred
    val_pts = validation_from_csv(bundle_id, zone_id, df)
    _enrich_prediction_metrics(pred, val_pts)
    pred["hours_ahead"] = hours_ahead
    return pred


@app.get("/api/forecast/validation-history")
def get_validation_history(bundle_id: str = DEFAULT_BUNDLE_ID, zone_id: int = 161):
    df = get_ml_dataframe_for_inference()
    pts = validation_from_csv(bundle_id, zone_id, df)
    return {"data": pts, "validation_metrics_by_model": _per_model_metrics_from_validation(pts)}


@app.get("/api/system/cluster-health")
def get_cluster_health():
    return build_dashboard_payload(DEFAULT_BUNDLE_ID, 161)["clusterHealth"]


@app.get("/api/system/streaming-status")
def get_streaming_status():
    return build_dashboard_payload(DEFAULT_BUNDLE_ID, 161)["streamingStatus"]


@app.websocket("/ws/dashboard")
async def websocket_dashboard(ws: WebSocket):
    await ws.accept()
    ids = registry.list_bundles()
    bundle_id = ids[0] if ids else DEFAULT_BUNDLE_ID
    zone_id = 161
    try:
        hello = {
            "type": "hello",
            "bundles": registry.list_bundles(),
            "bundle_catalog": _bundle_catalog(registry.list_bundles()),
            "source_meta": {
                **_static_source_meta(),
                "data_source": None,
                "ml_frame_rows": None,
                "ml_sample_time_range": None,
                "silver_row_cap": int(SILVER_ROW_CAP),
                "silver_ingest": None,
            },
        }
        await ws.send_json(jsonable_encoder(hello))
        while True:
            deadline = time.monotonic() + WS_TICK_SECONDS
            while time.monotonic() < deadline:
                try:
                    remaining = max(0.02, deadline - time.monotonic())
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=remaining)
                    msg = json.loads(raw)
                    if msg.get("type") == "config":
                        bundle_id = str(msg.get("bundle_id", bundle_id))
                        zone_id = int(msg.get("zone_id", zone_id))
                except asyncio.TimeoutError:
                    break
                except WebSocketDisconnect:
                    raise
                except Exception:
                    break
            payload = build_dashboard_payload(bundle_id, zone_id)
            await ws.send_json(jsonable_encoder(payload))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
