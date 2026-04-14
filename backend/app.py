from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import datetime, timedelta
from typing import List, Dict
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated Spark Data Mart - Pre-aggregated zones
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

def generate_zone_heatmap():
    """Simulate Spark aggregation output - trips by zone"""
    data = {}
    for zone_id, zone_name in TAXI_ZONES.items():
        # Simulate demand variability
        base_demand = random.randint(800, 3500)
        data[zone_id] = {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "trip_count": base_demand,
            "avg_fare": round(random.uniform(12.5, 45.0), 2),
            "density": round(base_demand / 100, 2)  # For color intensity
        }
    return list(data.values())

def generate_hourly_metrics():
    """Simulate Spark time-series aggregation"""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    metrics = []
    
    for i in range(24):
        hour = (base_time - timedelta(hours=23-i)).hour
        
        # Simulate peak hours (7-9am, 5-8pm)
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

@app.get("/api/overview/kpis")
def get_overview_kpis():
    """Executive KPIs - Simulates reading from pre-aggregated Data Mart"""
    current_trips = random.randint(148000, 162000)
    previous_trips = random.randint(135000, 145000)
    pct_change_trips = round(((current_trips - previous_trips) / previous_trips) * 100, 1)
    
    current_revenue = current_trips * random.uniform(17.5, 19.5)
    previous_revenue = previous_trips * random.uniform(17.5, 19.5)
    pct_change_revenue = round(((current_revenue - previous_revenue) / previous_revenue) * 100, 1)
    
    return {
        "total_trips": current_trips,
        "total_trips_change": pct_change_trips,
        "total_revenue": round(current_revenue, 2),
        "total_revenue_change": pct_change_revenue,
        "avg_trip_duration": round(random.uniform(14.2, 17.8), 1),
        "avg_fare": round(random.uniform(17.5, 19.5), 2),
        "spark_processing_time_ms": random.randint(120, 280),
        "data_mart_size_mb": round(random.uniform(145.5, 162.3), 1)
    }

@app.get("/api/overview/hourly-trends")
def get_hourly_trends():
    """24-hour metrics from Spark aggregation"""
    return {"data": generate_hourly_metrics()}

@app.get("/api/overview/zone-heatmap")
def get_zone_heatmap():
    """Zone-level aggregation from Spark GroupBy"""
    return {"data": generate_zone_heatmap()}

@app.get("/api/forecast/predict")
def run_forecast_prediction(zone_id: int = 161, hours_ahead: int = 1):
    """
    Simulate distributed ML model inference
    In production: Spark MLlib or TensorFlow on Spark
    """
    # Simulate model loading time
    processing_start = time.time()
    
    zone_name = TAXI_ZONES.get(zone_id, "Unknown Zone")
    base_demand = random.randint(180, 420)
    
    # Simulate slight model variance
    lstm_prediction = base_demand + random.randint(-25, 35)
    xgboost_prediction = base_demand + random.randint(-30, 40)
    
    # Ensemble average
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
    """Historical actual vs predicted for model validation"""
    base_time = datetime.now() - timedelta(hours=24)
    history = []
    
    for i in range(25):
        timestamp = base_time + timedelta(hours=i)
        actual = random.randint(280, 520)
        
        # Simulate model following actual with some lag/error
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
    """Spark cluster monitoring metrics"""
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
    """Kafka + Spark Structured Streaming simulation"""
    return {
        "kafka_topics": ["nyc-taxi-raw", "nyc-taxi-processed"],
        "messages_per_second": random.randint(2800, 4500),
        "processing_latency_ms": random.randint(85, 180),
        "batch_interval_seconds": 5,
        "offset_lag": random.randint(120, 850),
        "status": "RUNNING",
        "last_checkpoint": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)