"""
Configuration settings for Vehicle Predictive Maintenance System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [SRC_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODELS_DIR / "vehicle_maintenance_model.pkl"
TRAINING_DATA_PATH = BASE_DIR / "engine_data.csv"

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# Vehicle data API configuration
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
VEHICLE_API_URL = os.getenv("VEHICLE_API_URL", "https://api.example.com")
VEHICLE_API_KEY = os.getenv("VEHICLE_API_KEY", None)

# Backend API configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:3000")

# Scheduler configuration
DAILY_PREDICTION_HOUR = int(os.getenv("DAILY_PREDICTION_HOUR", "2"))  # 2 AM UTC
PREDICTION_BATCH_SIZE = int(os.getenv("PREDICTION_BATCH_SIZE", "10"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "subsample": 0.8
}

# Feature columns
FEATURE_COLUMNS = [
    'Engine rpm',
    'Lub oil pressure', 
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp',
    'Temperature_difference'
]

# Prediction thresholds
MAINTENANCE_PROBABILITY_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.6

# Maintenance estimation
BASE_MAINTENANCE_INTERVAL_DAYS = 30
MAX_MAINTENANCE_INTERVAL_DAYS = 90
MIN_MAINTENANCE_INTERVAL_DAYS = 1
