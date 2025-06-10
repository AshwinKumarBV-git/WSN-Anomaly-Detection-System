from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import numpy as np


class SensorReading(BaseModel):
    """
    Model for a single sensor reading
    """
    temperature: float = Field(..., description="Temperature reading in Celsius")
    motion: int = Field(..., description="Motion detection (0 or 1)")
    pulse: float = Field(..., description="Pulse reading in BPM")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the reading")
    sensor_id: Optional[str] = Field(None, description="ID of the sensor")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """
        Validate temperature is within a reasonable range
        """
        if v < -50 or v > 100:
            raise ValueError(f"Temperature {v}°C is outside reasonable range (-50°C to 100°C)")
        return v
    
    @validator('motion')
    def validate_motion(cls, v):
        """
        Validate motion is binary (0 or 1)
        """
        if v not in [0, 1]:
            raise ValueError(f"Motion value must be 0 or 1, got {v}")
        return v
    
    @validator('pulse')
    def validate_pulse(cls, v):
        """
        Validate pulse is within a reasonable range
        """
        if v < 0 or v > 300:
            raise ValueError(f"Pulse {v} BPM is outside reasonable range (0 to 300 BPM)")
        return v


class SensorWindow(BaseModel):
    """
    Model for a window of sensor readings
    """
    readings: List[SensorReading] = Field(..., description="List of sensor readings in the window")
    window_size: int = Field(..., description="Size of the window")
    
    @validator('readings')
    def validate_readings(cls, v, values):
        """
        Validate that the number of readings matches the window size
        """
        if 'window_size' in values and len(v) != values['window_size']:
            raise ValueError(f"Expected {values['window_size']} readings, got {len(v)}")
        return v


class PredictionRequest(BaseModel):
    """
    Model for a prediction request
    """
    # Single reading or window of readings
    data: Union[SensorReading, SensorWindow] = Field(..., description="Sensor data for prediction")
    
    # Optional configuration
    model_type: Optional[str] = Field("ensemble", description="Model type to use for prediction (sklearn, autoencoder, ensemble)")
    return_probabilities: Optional[bool] = Field(False, description="Whether to return class probabilities")
    return_features: Optional[bool] = Field(False, description="Whether to return extracted features")


class AnomalyPrediction(BaseModel):
    """
    Model for an anomaly prediction result
    """
    prediction: str = Field(..., description="Predicted class (normal or anomaly type)")
    confidence: float = Field(..., description="Confidence score for the prediction")
    timestamp: datetime = Field(..., description="Timestamp of the prediction")
    sensor_id: Optional[str] = Field(None, description="ID of the sensor")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    features: Optional[Dict[str, float]] = Field(None, description="Extracted features")
    
    class Config:
        json_encoders = {
            # Custom JSON encoder for numpy types
            np.float32: float,
            np.float64: float,
            np.int32: int,
            np.int64: int
        }


class BatchPredictionRequest(BaseModel):
    """
    Model for a batch prediction request
    """
    data: List[SensorReading] = Field(..., description="List of sensor readings for batch prediction")
    model_type: Optional[str] = Field("ensemble", description="Model type to use for prediction")
    return_probabilities: Optional[bool] = Field(False, description="Whether to return class probabilities")


class BatchPredictionResponse(BaseModel):
    """
    Model for a batch prediction response
    """
    predictions: List[AnomalyPrediction] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics of the predictions")


class ModelInfo(BaseModel):
    """
    Model for information about available models
    """
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model (sklearn, autoencoder, ensemble)")
    features_used: List[str] = Field(..., description="List of features used by the model")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics of the model")
    last_updated: datetime = Field(..., description="Last time the model was updated")


class SystemStatus(BaseModel):
    """
    Model for system status information
    """
    status: str = Field(..., description="Overall system status")
    models_loaded: Dict[str, bool] = Field(..., description="Status of model loading")
    api_version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")


def validate_sensor_data(data):
    """
    Validate sensor data and convert to appropriate format
    
    Args:
        data: Raw sensor data (dict, list, or DataFrame)
        
    Returns:
        Validated SensorReading or SensorWindow object
    """
    try:
        # Handle different input types
        if isinstance(data, dict):
            # Single reading as dictionary
            return SensorReading(**data)
        
        elif isinstance(data, list):
            # List of readings
            readings = [SensorReading(**reading) for reading in data]
            return SensorWindow(readings=readings, window_size=len(readings))
        
        else:
            # Try to convert from pandas DataFrame or other format
            try:
                # If it's a DataFrame, convert to dict records
                records = data.to_dict(orient='records')
                readings = [SensorReading(**record) for record in records]
                return SensorWindow(readings=readings, window_size=len(readings))
            except AttributeError:
                raise ValueError(f"Unsupported data type: {type(data)}")
    
    except Exception as e:
        raise ValueError(f"Data validation error: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Test with valid data
    valid_reading = {
        "temperature": 25.5,
        "motion": 1,
        "pulse": 72.0,
        "timestamp": datetime.now(),
        "sensor_id": "WSN001"
    }
    
    try:
        validated = SensorReading(**valid_reading)
        print("Valid reading:", validated)
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Test with invalid data
    invalid_reading = {
        "temperature": 150.0,  # Outside valid range
        "motion": 2,          # Not binary
        "pulse": -10.0,       # Outside valid range
        "timestamp": datetime.now(),
        "sensor_id": "WSN001"
    }
    
    try:
        validated = SensorReading(**invalid_reading)
        print("Valid reading:", validated)
    except Exception as e:
        print(f"Validation error: {e}")