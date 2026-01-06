from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
import os

from src.models.predictor import VehicleMaintenancePredictor, DataSourceType
from src.data.vehicle_data_client import VehicleDataClient
from src.data.gps_iot_client import GPSVehicleData
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

class GPSIoTData(BaseModel):
    """GPS IoT data model for API"""
    vehicle_name: str = Field(..., description="Vehicle name/ID from GPS IoT")
    vin: str = Field(..., description="Vehicle VIN")
    engine_status: Optional[int] = Field(None, description="Engine status (0=off, 1=on)")
    ignition_status: Optional[int] = Field(None, description="Ignition status (0=off, 1=on)")
    speed: Optional[float] = Field(None, description="Current speed in km/h")
    odometer_km: Optional[float] = Field(None, description="Odometer reading in km")
    last_position_latitude: Optional[float] = Field(None, description="Last position latitude")
    last_position_longitude: Optional[float] = Field(None, description="Last position longitude")
    last_position_timestamp: Optional[str] = Field(None, description="Last position timestamp")
    trip_count: Optional[int] = Field(None, description="Daily trip count")
    trip_total_km: Optional[float] = Field(None, description="Daily total distance in km")
    trip_duration: Optional[str] = Field(None, description="Daily trip duration")
    voltage: Optional[float] = Field(None, description="GPS device voltage")

class HybridData(BaseModel):
    """Hybrid data model combining sensor and GPS IoT data"""
    sensor_data: SensorData
    gps_data: Optional[GPSIoTData] = None

class PredictionRequest(BaseModel):
    sensor_data: SensorData

class GPSIoTPredictionRequest(BaseModel):
    gps_data: GPSIoTData

class HybridPredictionRequest(BaseModel):
    hybrid_data: HybridData

class BatchPredictionRequest(BaseModel):
    sensor_data_list: List[SensorData]

class GPSIoTBatchPredictionRequest(BaseModel):
    gps_data_list: List[GPSIoTData]

class HybridBatchPredictionRequest(BaseModel):
    hybrid_data_list: List[HybridData]

class PredictionResponse(BaseModel):
    vehicle_id: str
    timestamp: str
    maintenance_required: bool
    maintenance_probability: float
    estimated_days_remaining_before_maintenance: int
    model_confidence: float
    data_source: Optional[str] = None
    features_used: Optional[List[str]] = None
    gps_insights: Optional[Dict[str, Any]] = None

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
gps_predictor = None
hybrid_predictor = None
scheduler = None
use_mock_data = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
data_source_type = os.getenv("DATA_SOURCE_TYPE", "sensor").lower()
vehicle_api_url = os.getenv("VEHICLE_API_URL", "https://api.example.com")
vehicle_api_key = os.getenv("VEHICLE_API_KEY", None)
backend_api_url = os.getenv("BACKEND_API_URL", "http://localhost:3000")

# GPS IoT Configuration
gps_base_url = os.getenv("GPS_BASE_URL", "https://base_url/prod/prod")
gps_username = os.getenv("GPS_USERNAME", "user_name")
gps_password = os.getenv("GPS_PASSWORD", "password")
gps_cache_ttl = int(os.getenv("GPS_CACHE_TTL", "300"))

def get_predictor() -> VehicleMaintenancePredictor:
    """Dependency to get the predictor instance."""
    global predictor
    if predictor is None:
        data_source = DataSourceType.SENSOR
        predictor = VehicleMaintenancePredictor(data_source_type=data_source)
    return predictor

def get_gps_predictor() -> VehicleMaintenancePredictor:
    """Dependency to get the GPS IoT predictor instance."""
    global gps_predictor
    if gps_predictor is None:
        data_source = DataSourceType.GPS_IOT
        gps_predictor = VehicleMaintenancePredictor(data_source_type=data_source)
    return gps_predictor

def get_hybrid_predictor() -> VehicleMaintenancePredictor:
    """Dependency to get the hybrid predictor instance."""
    global hybrid_predictor
    if hybrid_predictor is None:
        data_source = DataSourceType.HYBRID
        hybrid_predictor = VehicleMaintenancePredictor(data_source_type=data_source)
    return hybrid_predictor

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

def get_data_source_type() -> DataSourceType:
    """Get configured data source type."""
    if data_source_type == "gps_iot":
        return DataSourceType.GPS_IOT
    elif data_source_type == "hybrid":
        return DataSourceType.HYBRID
    elif data_source_type == "sensor":
        return DataSourceType.SENSOR
    else:
        return DataSourceType.SENSOR

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
        config = {
            'data_source': data_source_type,
            'vehicle_count': 10
        }
        
        async with VehicleDataClient(config) as client:
            vehicles = await client.get_vehicle_list()
            return {"vehicles": vehicles}
        
    except Exception as e:
        logger.error(f"Error fetching vehicles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch vehicles: {str(e)}")

@app.get("/vehicles/{vehicle_id}/latest-data")
async def get_latest_vehicle_data(vehicle_id: str):
    """Get latest sensor data for a specific vehicle."""
    try:
        config = {
            'data_source': data_source_type,
            'vehicle_count': 10
        }
        
        async with VehicleDataClient(config) as client:
            data = await client.get_sensor_data_for_prediction(vehicle_id)
        
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
            "model_confidence": prediction_result["model_confidence"]
        }
        
        # Add sensor_data if available
        if "sensor_data" in prediction_result:
            payload["sensor_data"] = prediction_result["sensor_data"]
        
        # Add GPS insights if available
        if "gps_insights" in prediction_result:
            payload["gps_insights"] = prediction_result["gps_insights"]
        
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

# GPS IoT Prediction Endpoints
@app.post("/predict/gps-iot")
async def predict_maintenance_from_gps_iot(
    request: GPSIoTPredictionRequest,
    gps_predictor: VehicleMaintenancePredictor = Depends(get_gps_predictor)
):
    """Predict maintenance using GPS IoT data."""
    try:
        # Convert GPS IoT data to GPSVehicleData
        gps_data = GPSVehicleData(
            vehicle_name=request.gps_data.vehicle_name,
            vin=request.gps_data.vin,
            engine_status=request.gps_data.engine_status,
            ignition_status=request.gps_data.ignition_status,
            speed=request.gps_data.speed,
            odometer_km=request.gps_data.odometer_km,
            voltage=request.gps_data.voltage
        )
        
        # Add position data if available
        if request.gps_data.last_position_latitude and request.gps_data.last_position_longitude:
            from ..data.gps_iot_client import GPSPosition
            gps_data.last_position = GPSPosition(
                latitude=request.gps_data.last_position_latitude,
                longitude=request.gps_data.last_position_longitude,
                timestamp=datetime.fromisoformat(request.gps_data.last_position_timestamp.replace('Z', '+00:00')) if request.gps_data.last_position_timestamp else datetime.utcnow()
            )
        
        # Add trip data if available
        if request.gps_data.trip_count is not None:
            from ..data.gps_iot_client import GPSTripData
            gps_data.trips = GPSTripData(
                count=request.gps_data.trip_count,
                total_duration=request.gps_data.trip_duration or "0:00:00",
                total_km=request.gps_data.trip_total_km or 0.0
            )
        
        # Make prediction
        result = gps_predictor.predict_from_gps_iot(gps_data)
        
        # Send to backend (async, don't wait for response)
        asyncio.create_task(send_to_backend(result))
        
        # Create response
        response = PredictionResponse(
            vehicle_id=result["vehicle_id"],
            timestamp=result["timestamp"],
            maintenance_required=result["maintenance_required"],
            maintenance_probability=result["maintenance_probability"],
            estimated_days_remaining_before_maintenance=result["estimated_days_remaining_before_maintenance"],
            model_confidence=result["model_confidence"],
            data_source=result.get("data_source"),
            features_used=result.get("features_used"),
            gps_insights=result.get("gps_insights")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"GPS IoT prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"GPS IoT prediction failed: {str(e)}")

@app.post("/predict/hybrid")
async def predict_maintenance_hybrid(
    request: HybridPredictionRequest,
    hybrid_predictor: VehicleMaintenancePredictor = Depends(get_hybrid_predictor)
):
    """Predict maintenance using hybrid sensor + GPS IoT data."""
    try:
        # Convert sensor data
        sensor_data = request.hybrid_data.sensor_data.dict()
        
        # Convert GPS IoT data if available
        gps_data = None
        if request.hybrid_data.gps_data:
            gps_data = GPSVehicleData(
                vehicle_name=request.hybrid_data.gps_data.vehicle_name,
                vin=request.hybrid_data.gps_data.vin,
                engine_status=request.hybrid_data.gps_data.engine_status,
                ignition_status=request.hybrid_data.gps_data.ignition_status,
                speed=request.hybrid_data.gps_data.speed,
                odometer_km=request.hybrid_data.gps_data.odometer_km,
                voltage=request.hybrid_data.gps_data.voltage
            )
            
            # Add position data if available
            if request.hybrid_data.gps_data.last_position_latitude and request.hybrid_data.gps_data.last_position_longitude:
                from ..data.gps_iot_client import GPSPosition
                gps_data.last_position = GPSPosition(
                    latitude=request.hybrid_data.gps_data.last_position_latitude,
                    longitude=request.hybrid_data.gps_data.last_position_longitude,
                    timestamp=datetime.fromisoformat(request.hybrid_data.gps_data.last_position_timestamp.replace('Z', '+00:00')) if request.hybrid_data.gps_data.last_position_timestamp else datetime.utcnow()
                )
            
            # Add trip data if available
            if request.hybrid_data.gps_data.trip_count is not None:
                from ..data.gps_iot_client import GPSTripData
                gps_data.trips = GPSTripData(
                    count=request.hybrid_data.gps_data.trip_count,
                    total_duration=request.hybrid_data.gps_data.trip_duration or "0:00:00",
                    total_km=request.hybrid_data.gps_data.trip_total_km or 0.0
                )
        
        # Make prediction
        result = hybrid_predictor.predict_hybrid(sensor_data, gps_data)
        
        # Send to backend (async, don't wait for response)
        asyncio.create_task(send_to_backend(result))
        
        # Create response
        response = PredictionResponse(
            vehicle_id=result["vehicle_id"],
            timestamp=result["timestamp"],
            maintenance_required=result["maintenance_required"],
            maintenance_probability=result["maintenance_probability"],
            estimated_days_remaining_before_maintenance=result["estimated_days_remaining_before_maintenance"],
            model_confidence=result["model_confidence"],
            data_source=result.get("data_source"),
            features_used=result.get("features_used"),
            gps_insights=result.get("gps_insights")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid prediction failed: {str(e)}")

@app.post("/predict/gps-iot/batch")
async def predict_maintenance_gps_iot_batch(
    request: GPSIoTBatchPredictionRequest,
    gps_predictor: VehicleMaintenancePredictor = Depends(get_gps_predictor)
):
    """Predict maintenance for multiple vehicles using GPS IoT data."""
    try:
        # Convert GPS IoT data list
        gps_data_list = []
        for gps_request in request.gps_data_list:
            gps_data = GPSVehicleData(
                vehicle_name=gps_request.vehicle_name,
                vin=gps_request.vin,
                engine_status=gps_request.engine_status,
                ignition_status=gps_request.ignition_status,
                speed=gps_request.speed,
                odometer_km=gps_request.odometer_km,
                voltage=gps_request.voltage
            )
            
            # Add position data if available
            if gps_request.last_position_latitude and gps_request.last_position_longitude:
                from ..data.gps_iot_client import GPSPosition
                gps_data.last_position = GPSPosition(
                    latitude=gps_request.last_position_latitude,
                    longitude=gps_request.last_position_longitude,
                    timestamp=datetime.fromisoformat(gps_request.last_position_timestamp.replace('Z', '+00:00')) if gps_request.last_position_timestamp else datetime.utcnow()
                )
            
            # Add trip data if available
            if gps_request.trip_count is not None:
                from ..data.gps_iot_client import GPSTripData
                gps_data.trips = GPSTripData(
                    count=gps_request.trip_count,
                    total_duration=gps_request.trip_duration or "0:00:00",
                    total_km=gps_request.trip_total_km or 0.0
                )
            
            gps_data_list.append(gps_data)
        
        # Make batch prediction
        results = gps_predictor.predict_batch(gps_data_list)
        
        # Send to backend (async, don't wait for response)
        for result in results:
            if 'error' not in result:
                asyncio.create_task(send_to_backend(result))
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"GPS IoT batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"GPS IoT batch prediction failed: {str(e)}")

@app.get("/gps-iot/vehicles")
async def get_gps_iot_vehicles():
    """Get list of vehicles from GPS IoT system."""
    try:
        from src.data.gps_iot_client import GPSIoTDataClient
        
        config = {
            'base_url': gps_base_url,
            'username': gps_username,
            'password': gps_password,
            'cache_ttl': gps_cache_ttl
        }
        
        client = GPSIoTDataClient(config)
        vehicles = await client.get_vehicle_list()
        await client.close()
        return {"vehicles": vehicles}
        
    except Exception as e:
        logger.error(f"Error fetching GPS IoT vehicles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch GPS IoT vehicles: {str(e)}")

@app.get("/gps-iot/vehicles/{vehicle_name}/data")
async def get_gps_iot_vehicle_data(vehicle_name: str):
    """Get GPS IoT data for a specific vehicle."""
    try:
        from src.data.gps_iot_client import GPSIoTDataClient
        
        config = {
            'base_url': gps_base_url,
            'username': gps_username,
            'password': gps_password,
            'cache_ttl': gps_cache_ttl
        }
        
        client = GPSIoTDataClient(config)
        gps_data = await client.get_vehicle_data(vehicle_name)
        await client.close()
        
        # Convert to dict for JSON response
        response = {
            "vehicle_name": gps_data.vehicle_name,
            "vin": gps_data.vin,
            "engine_status": gps_data.engine_status,
            "ignition_status": gps_data.ignition_status,
            "speed": gps_data.speed,
            "odometer_km": gps_data.odometer_km,
            "last_position": {
                "latitude": gps_data.last_position.latitude,
                "longitude": gps_data.last_position.longitude,
                "timestamp": gps_data.last_position.timestamp.isoformat()
            } if gps_data.last_position else None,
            "trips": {
                "count": gps_data.trips.count,
                "total_duration": gps_data.trips.total_duration,
                "total_km": gps_data.trips.total_km
            } if gps_data.trips else None,
            "parking_events": gps_data.parking_events,
            "voltage": gps_data.voltage,
            "last_update": gps_data.last_update.isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching GPS IoT vehicle data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch GPS IoT vehicle data: {str(e)}")

@app.get("/data-sources")
async def get_data_sources():
    """Get available data sources and current configuration."""
    try:
        return {
            "current_data_source": data_source_type,
            "available_sources": ["sensor", "gps_iot", "hybrid"],
            "gps_iot_configured": bool(gps_username and gps_password),
            "endpoints": {
                "sensor": "/predict",
                "gps_iot": "/predict/gps-iot",
                "hybrid": "/predict/hybrid",
                "batch_sensor": "/predict/batch",
                "batch_gps_iot": "/predict/gps-iot/batch",
                "batch_hybrid": "/predict/hybrid/batch"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data sources: {str(e)}")
        # Don't raise the error - just log it and continue

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
