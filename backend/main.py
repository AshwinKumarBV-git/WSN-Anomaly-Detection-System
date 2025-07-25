from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime
import time
import os
import sys
import psutil
import uvicorn

# Import local modules
from services.predictor import predictor, predict_anomaly
from services.data_validator import PredictionRequest, BatchPredictionRequest, SystemStatus, ModelInfo
from utils.logger import WSNLogger

# Initialize logger
logger = WSNLogger(name="wsn_api")

# Initialize FastAPI app
app = FastAPI(
    title="WSN Anomaly Detection API",
    description="API for detecting anomalies in Wireless Sensor Networks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track API start time
start_time = time.time()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for logging API requests."""
    start_time_req = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time_req) * 1000
    logger.log_api_request(
        endpoint=request.url.path,
        method=request.method,
        request_data=None, # Avoid logging potentially large/sensitive data
        response_data=None,
        status_code=response.status_code,
        duration_ms=duration_ms
    )
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log unhandled errors."""
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}",
        {"error": str(exc)},
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "message": str(exc)}
    )

@app.get("/")
async def root():
    return {"message": "WSN Anomaly Detection API is running."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Endpoint to get the current status of the system."""
    models_loaded = {
        "sklearn": predictor.sklearn_model is not None,
        "autoencoder": predictor.autoencoder_model is not None,
        "feature_extractor": predictor.feature_extractor is not None
    }
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    status = SystemStatus(
        status="operational" if all(models_loaded.values()) else "degraded",
        models_loaded=models_loaded,
        api_version="1.0.0",
        uptime_seconds=time.time() - start_time,
        memory_usage_mb=memory_info.rss / (1024 * 1024),
        cpu_usage_percent=process.cpu_percent()
    )
    return status

@app.get("/models", response_model=List[ModelInfo])
async def get_models_info():
    """Endpoint to get information about the loaded models."""
    models_info = []
    if predictor.sklearn_model is not None:
        models_info.append(ModelInfo(
            model_name="sklearn_model",
            model_type=predictor.sklearn_model_type,
            features_used=["statistical", "frequency", "cross_sensor"],
            performance_metrics={"accuracy": 0.95}, # Placeholder
            last_updated=datetime.now() # Placeholder
        ))
    if predictor.autoencoder_model is not None:
        models_info.append(ModelInfo(
            model_name="tf_autoencoder",
            model_type="autoencoder",
            features_used=["statistical", "frequency", "cross_sensor"],
            performance_metrics={"reconstruction_error": 0.05}, # Placeholder
            last_updated=datetime.now() # Placeholder
        ))
    return models_info

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Endpoint for a single prediction."""
    try:
        result = predict_anomaly(
            sensor_data=request.data,
            model_type=request.model_type,
            return_probabilities=request.return_probabilities,
            return_features=request.return_features
        )
        logger.log_prediction(result)
        return result
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Endpoint for batch predictions."""
    try:
        result = predictor.batch_predict(
            data=request.data,
            model_type=request.model_type,
            return_probabilities=request.return_probabilities
        )
        logger.info("Batch prediction completed", result.get("summary", {}))
        return result
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulate")
async def simulate_data(num_samples: int = 100, include_anomalies: bool = False):
    """Endpoint to generate simulated sensor data."""
    try:
        # **FIX**: Correctly add the project root to sys.path to find the simulation module
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from simulation.simulate_data import generate_temperature_data, generate_motion_data, generate_pulse_data, merge_sensor_data
        
        start_time_sim = datetime.now()
        temp_df = generate_temperature_data(num_samples, start_time_sim)
        motion_df = generate_motion_data(num_samples, start_time_sim)
        pulse_df = generate_pulse_data(num_samples, start_time_sim)
        merged_df = merge_sensor_data(temp_df, motion_df, pulse_df)
        
        if include_anomalies and num_samples >= 50: # Ensure enough data for anomalies
            from simulation.inject_anomalies import inject_dos_attack
            merged_df = inject_dos_attack(merged_df, attack_duration_minutes=5)
        
        data = merged_df.to_dict(orient="records")
        return {"data": data, "count": len(data)}
        
    except ModuleNotFoundError:
        logger.error("Simulation module not found. Ensure the project structure is correct.", exc_info=True)
        raise HTTPException(status_code=500, detail="Simulation module not found on server.")
    except Exception as e:
        logger.error(f"Error in data simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
