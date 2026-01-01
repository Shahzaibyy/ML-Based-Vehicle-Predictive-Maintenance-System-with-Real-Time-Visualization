import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class VehicleDataClient:
    def __init__(self, api_base_url: str, api_key: Optional[str] = None):
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    async def get_vehicle_data(self, vehicle_id: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch sensor data for a specific vehicle."""
        try:
            params = {'vehicle_id': vehicle_id}
            if start_time:
                params['start_time'] = start_time.isoformat()
            if end_time:
                params['end_time'] = end_time.isoformat()
            
            url = f"{self.api_base_url}/vehicles/{vehicle_id}/sensor-data"
            
            async with self.session.get(url, params=params, headers=self._get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('sensor_data', [])
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise Exception(f"API request failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching vehicle data for {vehicle_id}: {e}")
            raise
    
    async def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Fetch all vehicles from the API."""
        try:
            url = f"{self.api_base_url}/vehicles"
            
            async with self.session.get(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('vehicles', [])
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise Exception(f"API request failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching all vehicles: {e}")
            raise
    
    async def get_latest_sensor_data(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest sensor data for a vehicle."""
        try:
            url = f"{self.api_base_url}/vehicles/{vehicle_id}/latest-data"
            
            async with self.session.get(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('sensor_data')
                elif response.status == 404:
                    logger.warning(f"No data found for vehicle {vehicle_id}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise Exception(f"API request failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching latest data for {vehicle_id}: {e}")
            raise
    
    async def stream_vehicle_data(self, vehicle_ids: List[str], callback):
        """Stream real-time vehicle data (if supported by API)."""
        try:
            url = f"{self.api_base_url}/vehicles/stream"
            params = {'vehicle_ids': ','.join(vehicle_ids)}
            
            async with self.session.ws_connect(url, params=params, headers=self._get_headers()) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await callback(data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
        
        except Exception as e:
            logger.error(f"Error in vehicle data streaming: {e}")
            raise

class MockVehicleDataClient:
    """Mock client for testing with dummy data."""
    
    def __init__(self, num_vehicles: int = 10):
        self.num_vehicles = num_vehicles
        self.vehicles = self._generate_vehicles()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _generate_vehicles(self) -> List[Dict[str, Any]]:
        vehicles = []
        for i in range(self.num_vehicles):
            vehicles.append({
                'vehicle_id': f'VEH-{i+1:03d}',
                'make': 'Toyota',
                'model': 'Camry',
                'year': 2020 + (i % 4),
                'status': 'active',
                'last_maintenance': (datetime.now() - timedelta(days=30*i)).isoformat()
            })
        return vehicles
    
    def _generate_sensor_data(self, vehicle_id: str) -> Dict[str, Any]:
        """Generate realistic sensor data for testing."""
        import random
        
        # Base values with some randomness
        base_rpm = 800 + random.randint(-200, 400)
        base_lub_pressure = 3.0 + random.uniform(-0.5, 0.5)
        base_fuel_pressure = 6.0 + random.uniform(-1.0, 1.0)
        base_coolant_pressure = 2.0 + random.uniform(-0.3, 0.3)
        base_lub_temp = 77.0 + random.uniform(-2.0, 2.0)
        base_coolant_temp = 78.0 + random.uniform(-3.0, 3.0)
        
        # Add some anomalies for testing
        if random.random() < 0.1:  # 10% chance of anomaly
            base_rpm = random.choice([base_rpm * 1.5, base_rpm * 0.7])
            base_coolant_temp += random.uniform(10, 30)
        
        temp_difference = base_coolant_temp - base_lub_temp
        
        return {
            'vehicle_id': vehicle_id,
            'timestamp': datetime.utcnow().isoformat(),
            'Engine_rpm': base_rpm,
            'Lub_oil_pressure': max(0.1, base_lub_pressure),
            'Fuel_pressure': max(0.1, base_fuel_pressure),
            'Coolant_pressure': max(0.1, base_coolant_pressure),
            'lub_oil_temp': base_lub_temp,
            'Coolant_temp': base_coolant_temp,
            'Temperature_difference': temp_difference
        }
    
    async def get_vehicle_data(self, vehicle_id: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Mock implementation - returns recent sensor data."""
        data_points = []
        for i in range(10):  # Return 10 data points
            sensor_data = self._generate_sensor_data(vehicle_id)
            # Adjust timestamp for historical data
            if i > 0:
                sensor_data['timestamp'] = (datetime.utcnow() - timedelta(minutes=i*5)).isoformat()
            data_points.append(sensor_data)
        
        return data_points
    
    async def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """Mock implementation - returns generated vehicles."""
        return self.vehicles
    
    async def get_latest_sensor_data(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Mock implementation - returns latest sensor data."""
        return self._generate_sensor_data(vehicle_id)
    
    async def stream_vehicle_data(self, vehicle_ids: List[str], callback):
        """Mock implementation - simulates streaming with periodic updates."""
        while True:
            for vehicle_id in vehicle_ids:
                data = self._generate_sensor_data(vehicle_id)
                await callback(data)
            await asyncio.sleep(5)  # Update every 5 seconds
