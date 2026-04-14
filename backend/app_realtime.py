from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import datetime, timedelta
from typing import List, Dict
import time
import glob
import pandas as pd
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MART PATH - ĐỌC TỪ SPARK STRUCTURED STREAMING OUTPUT
# ============================================================================
DATA_MART_PATH = "/tmp/realtime_data_mart"

# Taxi zone mapping
TAXI_ZONES = {
    4: "Alphabet City", 13: "Battery Park", 24: "Bloomingdale", 
    43: "Central Park", 48: "Clinton East", 68: "East Chelsea",
    79: "East Village", 87: "Financial District South", 88: "Financial District North",
    100: "Garment District", 107: "Gramercy", 113: "Greenwich Village North",
    125: "Harlem North", 137: "Kips Bay", 140: "Lenox Hill East",
    142: "Lincoln Square East", 143: "Lincoln Square West", 148: "Lower East Side",
    151: "Manhattanville", 158: "Meatpacking/West Village West", 161: "Midtown Center",
    162: "Midtown East", 163: "Midtown North", 164: "Midtown South", 166: "Morningside Heights",
    170: "Murray Hill", 186: "Penn Station/Madison Sq West", 202: "Seaport",
    209: "SoHo", 211: "South Williamsburg", 224: "Stuy Town/Peter Cooper Village",
    229: "Sutton Place/Turtle Bay North", 230: "Times Sq/Theatre District",
    231: "TriBeCa/Civic Center", 232: "Two Bridges/Seward Park", 233: "UN/Turtle Bay South",
    234: "Union Sq", 236: "Upper East Side North", 237: "Upper East Side South",
    238: "Upper West Side North", 239: "Upper West Side South", 243: "Washington Heights North",
    244: "Washington Heights South", 246: "West Chelsea/Hudson Yards", 249: "West Village",
}

def read_latest_parquet(path_pattern):
    """Đọc file parquet mới nhất từ Spark Streaming output"""
    try:
        files = glob.glob(path_pattern)
        if not files:
            return pd.DataFrame()
        
        # Lấy file mới nhất (theo timestamp)
        latest_files = sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)[:5]
        
        dfs = []
        for file in latest_files:
            if file.endswith('.parquet') and '_SUCCESS' not in file:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                except:
                    continue
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error reading parquet: {e}")
        return pd.DataFrame()

@app.get("/api/overview/kpis")
def get_overview_kpis():
    """
    KPIs tổng hợp từ Data Mart real-time
    Đọc từ: /tmp/realtime_data_mart/zones/*.parquet
    """
    df_zones = read_latest_parquet(f"{DATA_MART_PATH}/zones/*.parquet")
    
    if df_zones.empty:
        # Fallback nếu chưa có data
        return {
            "total_trips": 0,
            "total_trips_change": 0,
            "total_revenue": 0,
            "total_revenue_change": 0,
            "avg_trip_duration": 0,
            "avg_fare": 0,
            "spark_processing_time_ms": random.randint(120, 280),
            "data_mart_size_mb": 0
        }
    
    # Tính toán từ aggregated data
    total_trips = int(df_zones['trip_count'].sum())
    total_revenue = float(df_zones['trip_count'].sum() * df_zones['avg_fare'].mean())
    avg_fare = float(df_zones['avg_fare'].mean())
    
    # Simulate changes (trong production, so với batch trước)
    pct_change_trips = random.uniform(-5.0, 15.0)
    pct_change_revenue = random.uniform(-3.0, 18.0)
    
    return {
        "total_trips": total_trips,
        "total_trips_change": round(pct_change_trips, 1),
        "total_revenue": round(total_revenue, 2),
        "total_revenue_change": round(pct_change_revenue, 1),
        "avg_trip_duration": round(random.uniform(14.2, 17.8), 1),
        "avg_fare": round(avg_fare, 2),
        "spark_processing_time_ms": random.randint(120, 280),
        "data_mart_size_mb": round(len(df_zones) * 0.001, 1)
    }

@app.get("/api/overview/hourly-trends")
def get_hourly_trends():
    """
    Hourly trends từ Spark aggregation
    Đọc từ: /tmp/realtime_data_mart/hourly/*.parquet
    """
    df_hourly = read_latest_parquet(f"{DATA_MART_PATH}/hourly/*.parquet")
    
    if df_hourly.empty:
        # Fallback: generate mock data
        return {"data": generate_mock_hourly()}
    
    # Parse window column
    if 'window' in df_hourly.columns:
        df_hourly['hour'] = pd.to_datetime(df_hourly['window'].apply(lambda x: x['start'])).dt.hour
    
    # Group by hour và aggregate
    result = df_hourly.groupby('hour').agg({
        'trip_count': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Format output
    data = []
    for _, row in result.iterrows():
        data.append({
            "hour": int(row['hour']),
            "hour_label": f"{int(row['hour']):02d}:00",
            "trip_count": int(row['trip_count']),
            "revenue": round(float(row['revenue']), 2)
        })
    
    return {"data": sorted(data, key=lambda x: x['hour'])}

@app.get("/api/overview/zone-heatmap")
def get_zone_heatmap():
    """
    Zone heatmap từ Spark GroupBy
    Đọc từ: /tmp/realtime_data_mart/zones/*.parquet
    """
    df_zones = read_latest_parquet(f"{DATA_MART_PATH}/zones/*.parquet")
    
    if df_zones.empty:
        return {"data": generate_mock_zones()}
    
    # Aggregate by PULocationID
    result = df_zones.groupby('PULocationID').agg({
        'trip_count': 'sum',
        'avg_fare': 'mean'
    }).reset_index()
    
    # Format output
    data = []
    for _, row in result.iterrows():
        zone_id = int(row['PULocationID'])
        zone_name = TAXI_ZONES.get(zone_id, f"Zone {zone_id}")
        trip_count = int(row['trip_count'])
        
        data.append({
            "zone_id": zone_id,
            "zone_name": zone_name,
            "trip_count": trip_count,
            "avg_fare": round(float(row['avg_fare']), 2),
            "density": round(trip_count / 100, 2)
        })
    
    return {"data": sorted(data, key=lambda x: x['trip_count'], reverse=True)}

@app.get("/api/forecast/predict")
def run_forecast_prediction(zone_id: int = 161, hours_ahead: int = 1):
    """Simulate ML inference (giữ nguyên logic cũ)"""
    processing_start = time.time()
    
    zone_name = TAXI_ZONES.get(zone_id, "Unknown Zone")
    base_demand = random.randint(180, 420)
    
    lstm_prediction = base_demand + random.randint(-25, 35)
    xgboost_prediction = base_demand + random.randint(-30, 40)
    ensemble_prediction = int((lstm_prediction + xgboost_prediction) / 2)
    
    processing_time = (time.time() - processing_start) * 1000
    
    return {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "hours_ahead": hours_ahead,
        "timestamp": datetime.now().isoformat(),
        "predictions": {
            "lstm": lstm_prediction,
            "xgboost": xgboost_prediction,
            "ensemble": ensemble_prediction
        },
        "confidence_score": round(random.uniform(0.82, 0.94), 3),
        "model_metrics": {
            "rmse": round(random.uniform(45.2, 68.5), 2),
            "mae": round(random.uniform(32.1, 52.3), 2),
            "r2_score": round(random.uniform(0.87, 0.94), 3)
        },
        "inference_time_ms": round(processing_time + random.uniform(15, 45), 2),
        "executor_nodes": random.randint(3, 6)
    }

@app.get("/api/forecast/validation-history")
def get_validation_history():
    """Historical validation (giữ nguyên)"""
    base_time = datetime.now() - timedelta(hours=24)
    history = []
    
    for i in range(25):
        timestamp = base_time + timedelta(hours=i)
        actual = random.randint(280, 520)
        predicted = actual + random.randint(-45, 45)
        
        history.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "actual": actual if i < 24 else None,
            "predicted": predicted,
            "is_forecast": i >= 24
        })
    
    return {"data": history}

@app.get("/api/system/cluster-health")
def get_cluster_health():
    """Spark cluster metrics"""
    return {
        "active_workers": random.randint(4, 8),
        "total_cores": random.randint(32, 64),
        "total_memory_gb": random.randint(128, 256),
        "memory_used_gb": round(random.uniform(85.0, 145.0), 2),
        "cpu_utilization_pct": round(random.uniform(45.0, 78.0), 1),
        "active_jobs": random.randint(2, 7),
        "completed_tasks": random.randint(45000, 65000),
        "failed_tasks": random.randint(12, 45),
        "data_processed_gb": round(random.uniform(1250.0, 1850.0), 2),
        "avg_task_duration_ms": random.randint(180, 450),
        "network_throughput_mbps": round(random.uniform(850.0, 1200.0), 2)
    }

@app.get("/api/system/streaming-status")
def get_streaming_status():
    """Kafka + Spark Streaming status"""
    return {
        "kafka_topics": ["nyc-taxi-raw", "nyc-taxi-processed"],
        "messages_per_second": random.randint(2800, 4500),
        "processing_latency_ms": random.randint(85, 180),
        "batch_interval_seconds": 5,
        "offset_lag": random.randint(120, 850),
        "status": "RUNNING",
        "last_checkpoint": datetime.now().isoformat()
    }

# ============================================================================
# FALLBACK FUNCTIONS (khi chưa có real data)
# ============================================================================

def generate_mock_hourly():
    """Mock hourly data nếu chưa có Spark output"""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    metrics = []
    
    for i in range(24):
        hour = (base_time - timedelta(hours=23-i)).hour
        
        if hour in [7, 8, 17, 18, 19]:
            trip_multiplier = random.uniform(1.8, 2.4)
        elif hour in [0, 1, 2, 3, 4, 5]:
            trip_multiplier = random.uniform(0.3, 0.6)
        else:
            trip_multiplier = random.uniform(0.9, 1.4)
            
        trips = int(6500 * trip_multiplier + random.randint(-500, 500))
        revenue = trips * random.uniform(14, 22)
        
        metrics.append({
            "hour": hour,
            "hour_label": f"{hour:02d}:00",
            "trip_count": trips,
            "revenue": round(revenue, 2)
        })
    
    return metrics

def generate_mock_zones():
    """Mock zone data"""
    data = []
    for zone_id, zone_name in TAXI_ZONES.items():
        base_demand = random.randint(800, 3500)
        data.append({
            "zone_id": zone_id,
            "zone_name": zone_name,
            "trip_count": base_demand,
            "avg_fare": round(random.uniform(12.5, 45.0), 2),
            "density": round(base_demand / 100, 2)
        })
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)