import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import aiohttp

from src.models.predictor import VehicleMaintenancePredictor
from src.data.vehicle_data_client import VehicleDataClient

logger = logging.getLogger(__name__)

class PredictionScheduler:
    def __init__(self, predictor: VehicleMaintenancePredictor, backend_api_url: str,
                 use_mock_data: bool = True, vehicle_api_url: str = None, 
                 vehicle_api_key: str = None):
        self.predictor = predictor
        self.backend_api_url = backend_api_url
        self.use_mock_data = use_mock_data
        self.vehicle_api_url = vehicle_api_url
        self.vehicle_api_key = vehicle_api_key
        self.running = False
        self.daily_task = None
    
    async def start_daily_predictions(self):
        """Start the daily prediction scheduler."""
        if self.running:
            logger.warning("Daily predictions already running")
            return
        
        self.running = True
        logger.info("Starting daily prediction scheduler")
        
        # Schedule first run
        await self.schedule_next_run()
    
    async def stop(self):
        """Stop the daily prediction scheduler."""
        self.running = False
        if self.daily_task:
            self.daily_task.cancel()
            try:
                await self.daily_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Daily prediction scheduler stopped")
    
    async def schedule_next_run(self):
        """Schedule the next daily prediction run."""
        if not self.running:
            return
        
        # Calculate time until next run (2 AM daily)
        now = datetime.utcnow()
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        
        # If next run is in the past, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)
        
        delay = (next_run - now).total_seconds()
        
        logger.info(f"Scheduling next daily prediction run in {delay/3600:.1f} hours")
        
        # Schedule the task
        self.daily_task = asyncio.create_task(self._schedule_with_delay(delay))
    
    async def _schedule_with_delay(self, delay: float):
        """Schedule task with delay."""
        try:
            await asyncio.sleep(delay)
            if self.running:
                await self.run_daily_predictions()
                await self.schedule_next_run()
        except asyncio.CancelledError:
            logger.info("Daily prediction task cancelled")
    
    async def run_daily_predictions(self):
        """Run daily predictions for all vehicles."""
        logger.info("Starting daily predictions for all vehicles")
        
        try:
            # Get all vehicles
            vehicles = await self._get_all_vehicles()
            
            if not vehicles:
                logger.warning("No vehicles found for daily predictions")
                return
            
            # Process vehicles in batches to avoid overwhelming the system
            batch_size = 10
            total_vehicles = len(vehicles)
            processed = 0
            successful = 0
            failed = 0
            
            for i in range(0, total_vehicles, batch_size):
                batch = vehicles[i:i + batch_size]
                batch_results = await self._process_vehicle_batch(batch)
                
                processed += len(batch)
                successful += batch_results['successful']
                failed += batch_results['failed']
                
                logger.info(f"Processed {processed}/{total_vehicles} vehicles "
                           f"(Success: {successful}, Failed: {failed})")
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < total_vehicles:
                    await asyncio.sleep(0.1)
            
            logger.info(f"Daily predictions completed: {successful} successful, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error in daily predictions: {e}")
    
    async def _get_all_vehicles(self) -> List[str]:
        """Get list of all vehicles."""
        try:
            # Use the new VehicleDataClient interface
            config = {
                'data_source': 'mock' if self.use_mock_data else 'gps_iot',
                'vehicle_count': 50
            }
            
            client = VehicleDataClient(config)
            vehicles = await client.get_vehicle_list()
            await client.close()
            return vehicles
        
        except Exception as e:
            logger.error(f"Error fetching vehicles: {e}")
            return []
    
    async def _process_vehicle_batch(self, vehicles: List[str]) -> Dict[str, int]:
        """Process a batch of vehicles for predictions."""
        successful = 0
        failed = 0
        
        # Create tasks for concurrent processing
        tasks = []
        for vehicle_id in vehicles:
            task = asyncio.create_task(self._process_single_vehicle(vehicle_id))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Vehicle processing failed: {result}")
            else:
                successful += 1
        
        return {"successful": successful, "failed": failed}
    
    async def _process_single_vehicle(self, vehicle_id: str) -> bool:
        """Process a single vehicle for prediction."""
        
        try:
            # Get latest sensor data
            sensor_data = await self._get_latest_sensor_data(vehicle_id)
            
            if sensor_data is None:
                logger.warning(f"No sensor data available for vehicle {vehicle_id}")
                return False
            
            # Make prediction
            prediction = self.predictor.predict_single(sensor_data)
            
            # Send to backend
            success = await self._send_prediction_to_backend(prediction)
            
            if success:
                logger.debug(f"Successfully processed vehicle {vehicle_id}")
            else:
                logger.warning(f"Failed to send prediction for vehicle {vehicle_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing vehicle {vehicle_id}: {e}")
            return False
    
    async def _get_latest_sensor_data(self, vehicle_id: str) -> Dict[str, Any]:
        """Get latest sensor data for a vehicle."""
        try:
            # Use the new VehicleDataClient interface
            config = {
                'data_source': 'mock' if self.use_mock_data else 'gps_iot',
                'vehicle_count': 50
            }
            
            client = VehicleDataClient(config)
            sensor_data = await client.get_sensor_data_for_prediction(vehicle_id)
            await client.close()
            return sensor_data
        
        except Exception as e:
            logger.error(f"Error fetching sensor data for {vehicle_id}: {e}")
            return None
    
    async def _send_prediction_to_backend(self, prediction: Dict[str, Any]) -> bool:
        """Send prediction result to backend API."""
        try:
            payload = {
                "vehicle_id": prediction["vehicle_id"],
                "maintenance_required": prediction["maintenance_required"],
                "maintenance_probability": prediction["maintenance_probability"],
                "estimated_days_remaining_before_maintenance": prediction["estimated_days_remaining_before_maintenance"],
                "prediction_timestamp": prediction["timestamp"],
                "model_confidence": prediction["model_confidence"]
            }
            
            # Add sensor_data if available
            if "sensor_data" in prediction:
                payload["sensor_data"] = prediction["sensor_data"]
            
            # Add GPS insights if available
            if "gps_insights" in prediction:
                payload["gps_insights"] = prediction["gps_insights"]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_api_url}/api/vehicle/maintenance-prediction",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent prediction for vehicle {prediction['vehicle_id']}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send prediction to backend: {response.status} - {error_text}")
                        return False
        
        except Exception as e:
            logger.error(f"Error sending prediction to backend: {e}")
            return False

class ManualPredictionTrigger:
    """Utility for manually triggering predictions."""
    
    def __init__(self, scheduler: PredictionScheduler):
        self.scheduler = scheduler
    
    async def trigger_for_vehicle(self, vehicle_id: str) -> Dict[str, Any]:
        """Trigger prediction for a specific vehicle."""
        try:
            sensor_data = await self.scheduler._get_latest_sensor_data(vehicle_id)
            
            if sensor_data is None:
                return {
                    "success": False,
                    "message": f"No sensor data available for vehicle {vehicle_id}"
                }
            
            prediction = self.scheduler.predictor.predict_single(sensor_data)
            backend_success = await self.scheduler._send_prediction_to_backend(prediction)
            
            return {
                "success": backend_success,
                "prediction": prediction,
                "message": f"Prediction processed for vehicle {vehicle_id}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing vehicle {vehicle_id}: {str(e)}"
            }
    
    async def trigger_for_all_vehicles(self) -> Dict[str, Any]:
        """Trigger predictions for all vehicles."""
        try:
            await self.scheduler.run_daily_predictions()
            return {
                "success": True,
                "message": "Manual predictions triggered for all vehicles"
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error triggering manual predictions: {str(e)}"
            }
