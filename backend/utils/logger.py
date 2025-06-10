import logging
import os
import sys
from datetime import datetime
import json
from typing import Dict, Any, Optional, Union


class WSNLogger:
    """
    Custom logger for WSN Anomaly Detection System
    Provides structured logging with different levels and formats
    """
    
    def __init__(self, name="wsn_anomaly", log_level=logging.INFO, 
                 log_to_console=True, log_to_file=True, log_dir="../logs"):
        """
        Initialize the logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    def _format_structured_log(self, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a structured log message with optional data
        
        Args:
            message: Log message
            data: Additional data to include in the log
            
        Returns:
            Formatted log message
        """
        if data is None:
            return message
        
        try:
            # Format data as JSON string
            data_str = json.dumps(data, default=str)
            return f"{message} | {data_str}"
        except Exception as e:
            return f"{message} | Error formatting data: {str(e)}"
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log a debug message
        
        Args:
            message: Log message
            data: Additional data to include in the log
        """
        self.logger.debug(self._format_structured_log(message, data))
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log an info message
        
        Args:
            message: Log message
            data: Additional data to include in the log
        """
        self.logger.info(self._format_structured_log(message, data))
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log a warning message
        
        Args:
            message: Log message
            data: Additional data to include in the log
        """
        self.logger.warning(self._format_structured_log(message, data))
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None, exc_info=False):
        """
        Log an error message
        
        Args:
            message: Log message
            data: Additional data to include in the log
            exc_info: Whether to include exception info
        """
        self.logger.error(self._format_structured_log(message, data), exc_info=exc_info)
    
    def critical(self, message: str, data: Optional[Dict[str, Any]] = None, exc_info=True):
        """
        Log a critical message
        
        Args:
            message: Log message
            data: Additional data to include in the log
            exc_info: Whether to include exception info
        """
        self.logger.critical(self._format_structured_log(message, data), exc_info=exc_info)
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log a prediction result
        
        Args:
            prediction_data: Prediction data to log
        """
        # Extract basic info for the message
        prediction = prediction_data.get("prediction", "unknown")
        confidence = prediction_data.get("confidence", 0.0)
        sensor_id = prediction_data.get("sensor_id", "unknown")
        
        # Create message
        message = f"Prediction: {prediction} (confidence: {confidence:.4f}) for sensor {sensor_id}"
        
        # Log with appropriate level based on prediction
        if prediction == "normal":
            self.info(message, prediction_data)
        elif prediction == "unknown" or prediction == "error":
            self.warning(message, prediction_data)
        else:
            # This is an anomaly, log as warning
            self.warning(message, prediction_data)
    
    def log_api_request(self, endpoint: str, method: str, request_data: Any, 
                        response_data: Any = None, status_code: int = 200, 
                        duration_ms: Optional[float] = None):
        """
        Log an API request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            request_data: Request data
            response_data: Response data
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        log_data = {
            "endpoint": endpoint,
            "method": method,
            "request": request_data,
            "status_code": status_code
        }
        
        if response_data is not None:
            log_data["response"] = response_data
        
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        
        # Create message
        message = f"{method} {endpoint} - {status_code}"
        if duration_ms is not None:
            message += f" ({duration_ms:.2f}ms)"
        
        # Log with appropriate level based on status code
        if status_code < 400:
            self.info(message, log_data)
        elif status_code < 500:
            self.warning(message, log_data)
        else:
            self.error(message, log_data)
    
    def log_model_loading(self, model_name: str, success: bool, error_msg: Optional[str] = None):
        """
        Log model loading
        
        Args:
            model_name: Name of the model
            success: Whether loading was successful
            error_msg: Error message if loading failed
        """
        log_data = {
            "model_name": model_name,
            "success": success
        }
        
        if error_msg is not None:
            log_data["error"] = error_msg
        
        # Create message
        if success:
            message = f"Successfully loaded model: {model_name}"
            self.info(message, log_data)
        else:
            message = f"Failed to load model: {model_name}"
            self.error(message, log_data)
    
    def log_system_metrics(self, metrics: Dict[str, Union[float, str, int]]):
        """
        Log system metrics
        
        Args:
            metrics: Dictionary of system metrics
        """
        # Create message
        message = "System metrics"
        self.info(message, metrics)


# Create a default logger instance
default_logger = WSNLogger()


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = WSNLogger(name="wsn_test", log_level=logging.DEBUG)
    
    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message", {"extra": "data", "count": 42})
    logger.warning("This is a warning message")
    
    # Log a prediction
    prediction_data = {
        "prediction": "DoS",
        "confidence": 0.92,
        "sensor_id": "WSN001",
        "timestamp": datetime.now()
    }
    logger.log_prediction(prediction_data)
    
    # Log an API request
    logger.log_api_request(
        endpoint="/predict",
        method="POST",
        request_data={"temperature": 25.5, "motion": 1, "pulse": 72.0},
        response_data=prediction_data,
        status_code=200,
        duration_ms=15.3
    )
    
    print(f"Logs written to: {logger.log_file}")