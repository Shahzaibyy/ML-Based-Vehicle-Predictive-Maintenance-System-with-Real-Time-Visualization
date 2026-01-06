"""
Enhanced Vehicle Data Client with GPS IoT Integration - SOLID Principles
Supports both mock data and real GPS IoT device data
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from abc import ABC, abstractmethod

from .gps_iot_client import GPSVehicleData
from ..models.predictor import DataSourceType

logger = logging.getLogger(__name__)

class VehicleDataProvider(ABC):
    """Abstract vehicle data provider - Dependency Inversion Principle"""
    
    @abstractmethod
    async def get_vehicle_data(self, vehicle_id: str) -> Union[Dict, GPSVehicleData]:
        """Get vehicle data"""
        pass
    
    @abstractmethod
    async def get_vehicle_list(self) -> List[str]:
        """Get list of available vehicles"""
        pass

class MockVehicleDataProvider(VehicleDataProvider):
    """Mock vehicle data provider - Liskov Substitution Principle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vehicle_count = config.get('vehicle_count', 10)
        self.base_vehicles = [f"VEH-{i:03d}" for i in range(1, self.vehicle_count + 1)]
    
    async def get_vehicle_data(self, vehicle_id: str) -> Dict:
        """Generate mock vehicle sensor data"""
        import random
        
        # Base sensor values with realistic ranges
        base_rpm = random.uniform(600, 2000)
        base_lub_pressure = random.uniform(2.0, 5.0)
        base_fuel_pressure = random.uniform(5.0, 10.0)
        base_coolant_pressure = random.uniform(1.5, 3.5)
        base_lub_temp = random.uniform(75, 90)
        base_coolant_temp = random.uniform(80, 95)
        
        # Add some realistic variations
        rpm_variation = random.uniform(-200, 300)
        pressure_variation = random.uniform(-0.5, 0.5)
        temp_variation = random.uniform(-5, 8)
        
        sensor_data = {
            'vehicle_id': vehicle_id,
            'Engine_rpm': max(0, base_rpm + rpm_variation),
            'Lub_oil_pressure': max(0.1, base_lub_pressure + pressure_variation),
            'Fuel_pressure': max(0.1, base_fuel_pressure + pressure_variation),
            'Coolant_pressure': max(0.1, base_coolant_pressure + pressure_variation),
            'lub_oil_temp': base_lub_temp + temp_variation,
            'Coolant_temp': base_coolant_temp + temp_variation,
            'Temperature_difference': (base_coolant_temp + temp_variation) - (base_lub_temp + temp_variation)
        }
        
        logger.debug(f"Generated mock data for {vehicle_id}")
        return sensor_data
    
    async def get_vehicle_list(self) -> List[str]:
        """Get list of mock vehicles"""
        return self.base_vehicles.copy()

class GPSIoTVehicleDataProvider(VehicleDataProvider):
    """GPS IoT vehicle data provider - Liskov Substitution Principle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.gps_client = GPSIoTDataClient(config)
        self.config = config
    
    async def get_vehicle_data(self, vehicle_id: str) -> GPSVehicleData:
        """Get real GPS IoT data"""
        try:
            gps_data = await self.gps_client.get_vehicle_data(vehicle_id)
            logger.info(f"Retrieved GPS IoT data for {vehicle_id}")
            return gps_data
        except Exception as e:
            logger.error(f"Error getting GPS IoT data for {vehicle_id}: {e}")
            # Return minimal GPS data with error flag
            return GPSVehicleData(
                vehicle_name=vehicle_id,
                vin="UNKNOWN",
                last_update=datetime.utcnow()
            )
    
    async def get_vehicle_list(self) -> List[str]:
        """Get list of vehicles from GPS IoT"""
        try:
            return await self.gps_client.get_vehicle_list()
        except Exception as e:
            logger.error(f"Error getting vehicle list from GPS IoT: {e}")
            return []
    
    async def close(self):
        """Close GPS IoT client"""
        await self.gps_client.close()

class HybridVehicleDataProvider(VehicleDataProvider):
    """Hybrid vehicle data provider - combines mock and GPS IoT data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.mock_provider = MockVehicleDataProvider(config)
        self.gps_provider = GPSIoTVehicleDataProvider(config)
        self.fallback_to_mock = config.get('fallback_to_mock', True)
    
    async def get_vehicle_data(self, vehicle_id: str) -> Dict:
        """Get hybrid vehicle data - GPS IoT with mock fallback"""
        try:
            # Try GPS IoT first
            gps_data = await self.gps_provider.get_vehicle_data(vehicle_id)
            
            # Check if GPS data is valid
            if self._is_valid_gps_data(gps_data):
                # Convert GPS data to sensor-compatible format
                sensor_data = await self._convert_gps_to_sensor_format(gps_data)
                
                # Return combined data
                return {
                    'sensor_data': sensor_data,
                    'gps_data': gps_data,
                    'data_source': 'hybrid',
                    'primary_source': 'gps_iot'
                }
            else:
                raise ValueError("Invalid GPS data")
                
        except Exception as e:
            logger.warning(f"GPS IoT data failed for {vehicle_id}, using mock: {e}")
            
            if self.fallback_to_mock:
                # Fallback to mock data
                mock_data = await self.mock_provider.get_vehicle_data(vehicle_id)
                return {
                    **mock_data,
                    'data_source': 'hybrid',
                    'primary_source': 'mock',
                    'fallback_reason': str(e)
                }
            else:
                raise e
    
    async def get_vehicle_list(self) -> List[str]:
        """Get combined vehicle list"""
        try:
            # Try GPS IoT list first
            gps_vehicles = await self.gps_provider.get_vehicle_list()
            if gps_vehicles:
                return gps_vehicles
        except Exception as e:
            logger.warning(f"GPS IoT vehicle list failed, using mock: {e}")
        
        # Fallback to mock list
        return await self.mock_provider.get_vehicle_list()
    
    def _is_valid_gps_data(self, gps_data: GPSVehicleData) -> bool:
        """Check if GPS data is valid"""
        # Check for essential data
        if not gps_data.vin or gps_data.vin == "UNKNOWN":
            return False
        
        # Check if data is recent (within last 24 hours)
        if gps_data.last_update:
            time_diff = datetime.utcnow() - gps_data.last_update
            if time_diff.total_seconds() > 86400:  # 24 hours
                return False
        
        return True
    
    async def _convert_gps_to_sensor_format(self, gps_data: GPSVehicleData) -> Dict:
        """Convert GPS data to sensor-compatible format"""
        # This would use the GPS feature engineer, but for now create basic mapping
        sensor_data = {
            'vehicle_id': gps_data.vehicle_name,
            'Engine_rpm': 800.0,  # Default values
            'Lub_oil_pressure': 3.0,
            'Fuel_pressure': 6.0,
            'Coolant_pressure': 2.0,
            'lub_oil_temp': 80.0,
            'Coolant_temp': 82.0,
            'Temperature_difference': 2.0
        }
        
        # Adjust based on GPS data if available
        if gps_data.speed and gps_data.speed > 0:
            sensor_data['Engine_rpm'] = 800 + (gps_data.speed * 20)
        
        if gps_data.engine_status == 1:
            sensor_data['Coolant_pressure'] *= 1.1
            sensor_data['Coolant_temp'] *= 1.05
        
        return sensor_data
    
    async def close(self):
        """Close providers"""
        await self.gps_provider.close()

class VehicleDataClient:
    """Enhanced vehicle data client - Facade Pattern with SOLID Principles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        data_source_str = config.get('data_source', 'mock')
        
        logger.info(f"VehicleDataClient initialized with data_source string: '{data_source_str}'")
        
        # Convert string to DataSourceType enum
        if data_source_str == 'sensor':
            self.data_source_type = DataSourceType.SENSOR
        elif data_source_str == 'gps_iot':
            self.data_source_type = DataSourceType.GPS_IOT
        elif data_source_str == 'hybrid':
            self.data_source_type = DataSourceType.HYBRID
        elif data_source_str == 'mock':
            self.data_source_type = DataSourceType.SENSOR  # Use SENSOR for mock data
        else:
            self.data_source_type = DataSourceType.SENSOR  # Default fallback
        
        logger.info(f"Converted to DataSourceType enum: {self.data_source_type}")
        
        # Dependency Injection - SOLID Principle
        self.provider = self._create_provider(self.data_source_type)
        
        logger.info(f"Vehicle Data Client initialized with {self.data_source_type.value} data source")
    
    def _create_provider(self, data_source_type: DataSourceType) -> VehicleDataProvider:
        """Factory method to create appropriate provider - Factory Pattern"""
        logger.info(f"Creating provider for data source type: {data_source_type}")
        logger.info(f"Data source type value: {data_source_type.value}")
        
        if data_source_type == DataSourceType.SENSOR:
            logger.info("Creating MockVehicleDataProvider for SENSOR")
            return MockVehicleDataProvider(self.config)  # Use mock for sensor data
        elif data_source_type == DataSourceType.GPS_IOT:
            logger.info("Creating GPSIoTVehicleDataProvider")
            return GPSIoTVehicleDataProvider(self.config)
        elif data_source_type == DataSourceType.HYBRID:
            logger.info("Creating HybridVehicleDataProvider")
            return HybridVehicleDataProvider(self.config)
        else:
            logger.error(f"Unsupported data source type: {data_source_type}")
            logger.error(f"Data source type value: {data_source_type.value}")
            raise ValueError(f"Unsupported data source type: {data_source_type}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def get_vehicle_data(self, vehicle_id: str) -> Union[Dict, GPSVehicleData]:
        """Get vehicle data from configured source"""
        return await self.provider.get_vehicle_data(vehicle_id)
    
    async def get_vehicle_list(self) -> List[str]:
        """Get list of available vehicles"""
        return await self.provider.get_vehicle_list()
    
    async def get_multiple_vehicles_data(self, vehicle_ids: List[str]) -> List[Union[Dict, GPSVehicleData]]:
        """Get data for multiple vehicles concurrently"""
        tasks = [self.get_vehicle_data(vehicle_id) for vehicle_id in vehicle_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error getting data for {vehicle_ids[i]}: {result}")
                # Add error result
                valid_results.append({
                    'vehicle_id': vehicle_ids[i],
                    'error': str(result),
                    'data_source': self.data_source_type.value
                })
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def get_sensor_data_for_prediction(self, vehicle_id: str) -> Dict:
        """Get sensor-compatible data for prediction"""
        data = await self.get_vehicle_data(vehicle_id)
        
        # Handle different data types
        if isinstance(data, GPSVehicleData):
            # Convert GPS data to sensor format
            from ..feature_engineers.gps_feature_engineer import GPSFeatureEngineer
            feature_engineer = GPSFeatureEngineer()
            return feature_engineer.create_sensor_compatible_features(data)
        elif isinstance(data, dict):
            # Check if it's hybrid data
            if 'sensor_data' in data:
                return data['sensor_data']
            else:
                # It's already sensor data
                return data
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")
    
    async def get_gps_data_for_prediction(self, vehicle_id: str) -> Optional[GPSVehicleData]:
        """Get GPS data for prediction"""
        data = await self.get_vehicle_data(vehicle_id)
        
        if isinstance(data, GPSVehicleData):
            return data
        elif isinstance(data, dict) and 'gps_data' in data:
            return data['gps_data']
        else:
            return None
    
    async def get_hybrid_data_for_prediction(self, vehicle_id: str) -> Dict:
        """Get hybrid data for prediction"""
        data = await self.get_vehicle_data(vehicle_id)
        
        if isinstance(data, dict) and 'sensor_data' in data and 'gps_data' in data:
            return data
        else:
            # Create hybrid data from single source
            if isinstance(data, GPSVehicleData):
                from ..feature_engineers.gps_feature_engineer import GPSFeatureEngineer
                feature_engineer = GPSFeatureEngineer()
                sensor_data = feature_engineer.create_sensor_compatible_features(data)
                
                return {
                    'sensor_data': sensor_data,
                    'gps_data': data,
                    'data_source': 'hybrid',
                    'primary_source': 'gps_iot'
                }
            else:
                # It's sensor data, create minimal GPS data
                return {
                    'sensor_data': data,
                    'gps_data': None,
                    'data_source': 'hybrid',
                    'primary_source': 'mock'
                }
    
    async def close(self):
        """Close the data client"""
        if hasattr(self.provider, 'close'):
            await self.provider.close()
        logger.info("Vehicle Data Client closed")
