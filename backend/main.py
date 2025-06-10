from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import os
import sys
import psutil
import uvicorn

# Import local modules
from services.predictor import predict_anomaly, AnomalyPredictor
from services.data_validator import SensorReading, PredictionRequest, BatchPredictionRequest, SystemStatus, ModelInfo
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
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize predictor
predictor = AnomalyPredictor()

# Track API start time
start_time = time.time()


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Start timer
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log request
    logger.log_api_request(
        endpoint=request.url.path,
        method=request.method,
        request_data=None,  # We don't log request body for privacy/security
        response_data=None,  # We don't log response body for privacy/security
        status_code=response.status_code,
        duration_ms=duration_ms
    )
    
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the exception
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}",
        {"error": str(exc)},
        exc_info=True
    )
    
    # Return error response
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )


# Root endpoint
@app.get("/")
async def root():
    return {"message": "WSN Anomaly Detection API", "status": "running"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}


# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    # Check if models are loaded
    models_loaded = {
        "sklearn": predictor.sklearn_model is not None,
        "autoencoder": predictor.autoencoder_model is not None,
        "feature_extractor": predictor.feature_extractor is not None
    }
    
    # Get system metrics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Create status response
    status = SystemStatus(
        status="operational" if any(models_loaded.values()) else "degraded",
        models_loaded=models_loaded,
        api_version="1.0.0",
        uptime_seconds=time.time() - start_time,
        memory_usage_mb=memory_info.rss / (1024 * 1024),
        cpu_usage_percent=process.cpu_percent()
    )
    
    # Log system metrics
    logger.log_system_metrics(status.dict())
    
    return status


# Model info endpoint
@app.get("/models", response_model=List[ModelInfo])
async def get_models_info():
    models_info = []
    
    # Add sklearn model info if available
    if predictor.sklearn_model is not None:
        sklearn_info = ModelInfo(
            model_name="sklearn_model",
            model_type="sklearn",
            features_used=["statistical", "frequency", "cross_sensor"],
            performance_metrics={
                "accuracy": 0.95,  # Placeholder values
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.93
            },
            last_updated=datetime.now()  # Placeholder
        )
        models_info.append(sklearn_info)
    
    # Add autoencoder model info if available
    if predictor.autoencoder_model is not None:
        autoencoder_info = ModelInfo(
            model_name="tf_autoencoder",
            model_type="autoencoder",
            features_used=["statistical", "frequency", "cross_sensor"],
            performance_metrics={
                "reconstruction_error": 0.05,  # Placeholder values
                "anomaly_detection_rate": 0.92,
                "false_positive_rate": 0.08
            },
            last_updated=datetime.now()  # Placeholder
        )
        models_info.append(autoencoder_info)
    
    return models_info


# Single prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        # Extract parameters
        data = request.data
        model_type = request.model_type
        return_probabilities = request.return_probabilities
        return_features = request.return_features
        
        # Make prediction
        if isinstance(data, SensorReading):
            # Single reading
            result = predict_anomaly(
                sensor_data=data,
                model_type=model_type,
                return_probabilities=return_probabilities,
                return_features=return_features
            )
        else:
            # Window of readings
            # Convert to DataFrame for processing
            import pandas as pd
            readings_df = pd.DataFrame([r.dict() for r in data.readings])
            
            # Use predictor directly for window
            result = predictor.predict(
                data=readings_df,
                model_type=model_type,
                return_probabilities=return_probabilities,
                return_features=return_features
            )
        
        # Log prediction in background
        background_tasks.add_task(logger.log_prediction, result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    try:
        # Extract parameters
        data = request.data
        model_type = request.model_type
        return_probabilities = request.return_probabilities
        
        # Convert to list of dicts
        data_dicts = [item.dict() for item in data]
        
        # Make batch prediction
        result = predictor.batch_predict(
            data=data_dicts,
            model_type=model_type,
            return_probabilities=return_probabilities
        )
        
        # Log summary
        logger.info("Batch prediction completed", result["summary"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Simulate data endpoint (for testing)
@app.get("/simulate")
async def simulate_data(num_samples: int = 1, include_anomalies: bool = False):
    try:
        # Import simulation modules
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation"))
        from simulation.simulate_data import generate_temperature_data, generate_motion_data, generate_pulse_data, merge_sensor_data
        
        # Generate data
        start_time = datetime.now()
        temp_df = generate_temperature_data(num_samples, start_time)
        motion_df = generate_motion_data(num_samples, start_time)
        pulse_df = generate_pulse_data(num_samples, start_time)
        
        # Merge data
        merged_df = merge_sensor_data(temp_df, motion_df, pulse_df)
        
        # Inject anomalies if requested
        if include_anomalies and num_samples >= 10:
            from simulation.inject_anomalies import inject_dos_attack
            merged_df = inject_dos_attack(merged_df, attack_duration_minutes=1)
        
        # Convert to list of dicts
        data = merged_df.to_dict(orient="records")
        
        return {"data": data, "count": len(data)}
    
    except Exception as e:
        logger.error(f"Error in data simulation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    # Create required directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)