from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
import os

from src.models.predictor import VehicleMaintenancePredictor
from src.data.vehicle_data_client import VehicleDataClient, MockVehicleDataClient
from src.utils.scheduler import PredictionScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class SensorData(BaseModel):
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    Engine_rpm: float = Field(..., ge=0, le=5000, description="Engine RPM")
    Lub_oil_pressure: float = Field(..., ge=0, le=10, description="Lubricating oil pressure")
    Fuel_pressure: float = Field(..., ge=0, le=30, description="Fuel pressure")
    Coolant_pressure: float = Field(..., ge=0, le=10, description="Coolant pressure")
    lub_oil_temp: float = Field(..., ge=-50, le=150, description="Lubricating oil temperature")
    Coolant_temp: float = Field(..., ge=-50, le=200, description="Coolant temperature")
    Temperature_difference: float = Field(..., ge=-100, le=100, description="Temperature difference")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of sensor reading")

class PredictionRequest(BaseModel):
    sensor_data: SensorData

class BatchPredictionRequest(BaseModel):
    sensor_data_list: List[SensorData]

class PredictionResponse(BaseModel):
    vehicle_id: str
    timestamp: str
    maintenance_required: bool
    maintenance_probability: float
    estimated_days_remaining_before_maintenance: int
    model_confidence: float

class BackendResponse(BaseModel):
    success: bool
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Predictive Maintenance API",
    description="ML-based predictive maintenance service for vehicle fleets",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
scheduler = None
use_mock_data = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
vehicle_api_url = os.getenv("VEHICLE_API_URL", "https://api.example.com")
vehicle_api_key = os.getenv("VEHICLE_API_KEY", None)
backend_api_url = os.getenv("BACKEND_API_URL", "http://localhost:3000")

def get_predictor() -> VehicleMaintenancePredictor:
    """Dependency to get the predictor instance."""
    global predictor
    if predictor is None:
        predictor = VehicleMaintenancePredictor()
    return predictor

def get_scheduler() -> PredictionScheduler:
    """Dependency to get the scheduler instance."""
    global scheduler
    if scheduler is None:
        scheduler = PredictionScheduler(
            predictor=get_predictor(),
            backend_api_url=backend_api_url,
            use_mock_data=use_mock_data,
            vehicle_api_url=vehicle_api_url,
            vehicle_api_key=vehicle_api_key
        )
    return scheduler

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Vehicle Predictive Maintenance API")
    
    # Initialize predictor
    get_predictor()
    
    # Start daily prediction scheduler
    sched = get_scheduler()
    asyncio.create_task(sched.start_daily_predictions())
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Vehicle Predictive Maintenance API")
    
    if scheduler:
        await scheduler.stop()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Vehicle Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "mock_data": use_mock_data
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        # Test model availability
        importance = predictor.get_feature_importance()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "features": list(importance.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/predict")
async def predict_maintenance(
    request: PredictionRequest,
    predictor: VehicleMaintenancePredictor = Depends(get_predictor)
):
    """Predict maintenance for a single vehicle."""
    try:
        # Convert Pydantic model to dict
        sensor_data = request.sensor_data.dict()
        
        # Validate sensor data
        is_valid, errors = predictor.validate_sensor_data(sensor_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid sensor data: {errors}")
        
        # Make prediction
        result = predictor.predict_single(sensor_data)
        
        # Send to backend (async, don't wait for response)
        asyncio.create_task(send_to_backend(result))
        
        # Create response with only the fields expected by PredictionResponse
        response = PredictionResponse(
            vehicle_id=result["vehicle_id"],
            timestamp=result["timestamp"],
            maintenance_required=result["maintenance_required"],
            maintenance_probability=result["maintenance_probability"],
            estimated_days_remaining_before_maintenance=result["estimated_days_remaining_before_maintenance"],
            model_confidence=result["model_confidence"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_maintenance_batch(
    request: BatchPredictionRequest,
    predictor: VehicleMaintenancePredictor = Depends(get_predictor)
):
    """Predict maintenance for multiple vehicles."""
    try:
        # Convert Pydantic models to dicts
        sensor_data_list = [data.dict() for data in request.sensor_data_list]
        
        # Validate all sensor data
        for i, sensor_data in enumerate(sensor_data_list):
            is_valid, errors = predictor.validate_sensor_data(sensor_data)
            if not is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid sensor data at index {i}: {errors}"
                )
        
        # Make predictions
        results = predictor.predict_batch(sensor_data_list)
        
        # Send to backend (async, don't wait for response)
        for result in results:
            if 'error' not in result:
                asyncio.create_task(send_to_backend(result))
        
        # Filter out error results and create proper response objects
        valid_results = []
        for result in results:
            if 'error' not in result:
                response = PredictionResponse(
                    vehicle_id=result["vehicle_id"],
                    timestamp=result["timestamp"],
                    maintenance_required=result["maintenance_required"],
                    maintenance_probability=result["maintenance_probability"],
                    estimated_days_remaining_before_maintenance=result["estimated_days_remaining_before_maintenance"],
                    model_confidence=result["model_confidence"]
                )
                valid_results.append(response)
        
        return valid_results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/vehicles")
async def get_vehicles():
    """Get list of all vehicles."""
    try:
        if use_mock_data:
            async with MockVehicleDataClient() as client:
                vehicles = await client.get_all_vehicles()
        else:
            async with VehicleDataClient(vehicle_api_url, vehicle_api_key) as client:
                vehicles = await client.get_all_vehicles()
        
        return {"vehicles": vehicles}
        
    except Exception as e:
        logger.error(f"Error fetching vehicles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch vehicles: {str(e)}")

@app.get("/vehicles/{vehicle_id}/latest-data")
async def get_latest_vehicle_data(vehicle_id: str):
    """Get latest sensor data for a specific vehicle."""
    try:
        if use_mock_data:
            async with MockVehicleDataClient() as client:
                data = await client.get_latest_sensor_data(vehicle_id)
        else:
            async with VehicleDataClient(vehicle_api_url, vehicle_api_key) as client:
                data = await client.get_latest_sensor_data(vehicle_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for vehicle {vehicle_id}")
        
        return {"sensor_data": data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching vehicle data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch vehicle data: {str(e)}")

@app.post("/trigger-daily-predictions")
async def trigger_daily_predictions(
    background_tasks: BackgroundTasks,
    scheduler: PredictionScheduler = Depends(get_scheduler)
):
    """Manually trigger daily predictions for all vehicles."""
    try:
        background_tasks.add_task(scheduler.run_daily_predictions)
        
        return {
            "success": True,
            "message": "Daily predictions triggered in background"
        }
        
    except Exception as e:
        logger.error(f"Error triggering daily predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger predictions: {str(e)}")

@app.get("/model/features")
async def get_model_features(predictor: VehicleMaintenancePredictor = Depends(get_predictor)):
    """Get model feature importance."""
    try:
        importance = predictor.get_feature_importance()
        
        return {
            "features": importance,
            "total_features": len(importance)
        }
        
    except Exception as e:
        logger.error(f"Error getting model features: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model features: {str(e)}")

async def send_to_backend(prediction_result: Dict[str, Any]):
    """Send prediction results to MERN backend."""
    try:
        import aiohttp
        
        payload = {
            "vehicle_id": prediction_result["vehicle_id"],
            "maintenance_required": prediction_result["maintenance_required"],
            "maintenance_probability": prediction_result["maintenance_probability"],
            "estimated_days_remaining_before_maintenance": prediction_result["estimated_days_remaining_before_maintenance"],
            "prediction_timestamp": prediction_result["timestamp"],
            "sensor_data": prediction_result["sensor_data"],
            "model_confidence": prediction_result["model_confidence"]
        }
        
        # Only try to send if backend URL is configured and not localhost
        if backend_api_url and not backend_api_url.startswith("http://localhost"):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{backend_api_url}/api/vehicle/maintenance-prediction",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent prediction for vehicle {prediction_result['vehicle_id']}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send prediction to backend: {response.status} - {error_text}")
        else:
            logger.info(f"Backend integration disabled or using localhost - skipping send for vehicle {prediction_result['vehicle_id']}")
    
    except Exception as e:
        logger.error(f"Error sending prediction to backend: {e}")
        # Don't raise the error - just log it and continue

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
