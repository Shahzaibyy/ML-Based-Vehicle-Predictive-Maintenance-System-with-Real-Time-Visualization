"""
GPS IoT Data Client - Single Responsibility Principle
Handles all GPS IoT device communication and data fetching
"""

import aiohttp
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class GPSDataType(Enum):
    """GPS IoT data types - Single Responsibility for data categorization"""
    ENGINE_STATUS = "engineStatus"
    IGNITION = "ignition"
    LAST_POSITION = "lastPos"
    SPEED = "speed"
    ODOMETER = "odometros"
    TRIPS = "recorridos"
    PARKING = "estacionamientos"
    CONSUMPTION = "consumos"
    VOLTAGE = "voltage"
    MOVEMENT = "sinMov"

@dataclass
class GPSPosition:
    """GPS position data - Single Responsibility for location data"""
    latitude: float
    longitude: float
    timestamp: datetime
    
    @classmethod
    def from_api_response(cls, response_data: Dict) -> 'GPSPosition':
        """Factory method to create GPSPosition from API response"""
        return cls(
            latitude=float(response_data['y']),
            longitude=float(response_data['x']),
            timestamp=datetime.fromisoformat(response_data['t'].replace('Z', '+00:00'))
        )

@dataclass
class GPSTripData:
    """GPS trip data - Single Responsibility for trip information"""
    count: int
    total_duration: str  # Format: "H:MM:SS"
    total_km: float
    
    @classmethod
    def from_api_response(cls, response_data: Dict) -> 'GPSTripData':
        """Factory method to create GPSTripData from API response"""
        return cls(
            count=int(response_data['count']),
            total_duration=response_data['totalDuration'],
            total_km=float(response_data['totalKm'].replace(' km', ''))
        )

@dataclass
class GPSVehicleData:
    """Complete GPS vehicle data - Single Responsibility for vehicle data aggregation"""
    vehicle_name: str
    vin: str
    engine_status: Optional[int] = None
    ignition_status: Optional[int] = None
    speed: Optional[float] = None
    odometer_km: Optional[float] = None
    last_position: Optional[GPSPosition] = None
    trips: Optional[GPSTripData] = None
    parking_events: Optional[List[Dict]] = None
    consumption_data: Optional[Dict] = None
    voltage: Optional[float] = None
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.utcnow()

class GPSIoTAuthenticator:
    """Handles GPS IoT authentication - Single Responsibility Principle"""
    
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self._session_token = None
        self._token_expiry = None
    
    async def authenticate(self, session: aiohttp.ClientSession) -> bool:
        """Authenticate with GPS IoT system"""
        try:
            auth_payload = {
                "body": {
                    "user": self.username,
                    "password": self.password
                }
            }
            
            async with session.post(f"{self.base_url}/auth", json=auth_payload) as response:
                if response.status == 200:
                    # Store session token (implementation depends on actual API response)
                    self._session_token = "dummy_token"  # Replace with actual token extraction
                    self._token_expiry = datetime.utcnow() + timedelta(hours=1)
                    logger.info("GPS IoT authentication successful")
                    return True
                else:
                    logger.error(f"GPS IoT authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"GPS IoT authentication error: {e}")
            return False
    
    def is_token_valid(self) -> bool:
        """Check if authentication token is still valid"""
        return (self._session_token is not None and 
                self._token_expiry is not None and 
                datetime.utcnow() < self._token_expiry)

class GPSDataFetcher(ABC):
    """Abstract base class for GPS data fetchers - Dependency Inversion Principle"""
    
    @abstractmethod
    async def fetch_data(self, vehicle_name: str, data_type: GPSDataType) -> Dict[str, Any]:
        """Fetch specific GPS data type for a vehicle"""
        pass

class GPSIoTDataFetcher(GPSDataFetcher):
    """Concrete implementation of GPS data fetcher - Liskov Substitution Principle"""
    
    def __init__(self, base_url: str, authenticator: GPSIoTAuthenticator):
        self.base_url = base_url
        self.authenticator = authenticator
        self.session = None
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self.authenticator.is_token_valid():
            if not self.session:
                self.session = aiohttp.ClientSession()
            return await self.authenticator.authenticate(self.session)
        return True
    
    async def fetch_data(self, vehicle_name: str, data_type: GPSDataType) -> Dict[str, Any]:
        """Fetch specific GPS data type for a vehicle"""
        if not await self._ensure_authenticated():
            raise Exception("GPS IoT authentication failed")
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            payload = {"body": {"reportType": data_type.value}}
            
            async with self.session.post(f"{self.base_url}/reports", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    parsed_data = json.loads(data['body'])
                    
                    # Extract vehicle-specific data
                    vehicle_data = parsed_data.get('parsedData', {}).get(vehicle_name, {})
                    return vehicle_data
                else:
                    logger.error(f"Failed to fetch {data_type.value}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching {data_type.value} for {vehicle_name}: {e}")
            return {}
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()

class GPSDataCache:
    """GPS data caching - Single Responsibility Principle"""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl_seconds):
                return data
            else:
                del self.cache[key]  # Remove expired entry
        return None
    
    def set(self, key: str, data: Dict):
        """Set data in cache"""
        self.cache[key] = (data, datetime.utcnow())
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()

class GPSIoTDataClient:
    """Main GPS IoT data client - Facade Pattern for simplified interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'https://base_url/prod/prod')
        self.username = config.get('username', 'user_name')
        self.password = config.get('password', 'password')
        self.cache_ttl = config.get('cache_ttl', 300)
        
        # Dependency Injection - SOLID Principle
        self.authenticator = GPSIoTAuthenticator(self.base_url, self.username, self.password)
        self.data_fetcher = GPSIoTDataFetcher(self.base_url, self.authenticator)
        self.cache = GPSDataCache(self.cache_ttl)
        
        logger.info("GPS IoT Data Client initialized")
    
    async def get_vehicle_data(self, vehicle_name: str, use_cache: bool = True) -> GPSVehicleData:
        """Get comprehensive vehicle data from GPS IoT"""
        cache_key = f"vehicle_data_{vehicle_name}"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached data for {vehicle_name}")
                return GPSVehicleData(**cached_data)
        
        try:
            # Fetch all relevant data types in parallel
            tasks = [
                self._fetch_engine_status(vehicle_name),
                self._fetch_ignition(vehicle_name),
                self._fetch_speed(vehicle_name),
                self._fetch_odometer(vehicle_name),
                self._fetch_last_position(vehicle_name),
                self._fetch_trips(vehicle_name),
                self._fetch_parking(vehicle_name),
                self._fetch_consumption(vehicle_name),
                self._fetch_voltage(vehicle_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            vehicle_data = GPSVehicleData(
                vehicle_name=vehicle_name,
                vin=results[0].get('VIN', 'Unknown'),
                engine_status=self._safe_int(results[0].get('engineStatus')),
                ignition_status=self._safe_int(results[1].get('ignition')),
                speed=self._extract_speed(results[2].get('speed')),
                odometer_km=self._extract_odometer(results[3].get('odo')),
                last_position=self._extract_position(results[4]),
                trips=self._extract_trips(results[5]),
                parking_events=results[6].get('events', []),
                consumption_data=results[7],
                voltage=self._safe_float(results[8].get('voltage'))
            )
            
            # Cache the result
            self.cache.set(cache_key, vehicle_data.__dict__)
            
            logger.info(f"Successfully fetched GPS data for {vehicle_name}")
            return vehicle_data
            
        except Exception as e:
            logger.error(f"Error fetching GPS data for {vehicle_name}: {e}")
            raise
    
    async def _fetch_engine_status(self, vehicle_name: str) -> Dict:
        """Fetch engine status data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.ENGINE_STATUS)
    
    async def _fetch_ignition(self, vehicle_name: str) -> Dict:
        """Fetch ignition data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.IGNITION)
    
    async def _fetch_speed(self, vehicle_name: str) -> Dict:
        """Fetch speed data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.SPEED)
    
    async def _fetch_odometer(self, vehicle_name: str) -> Dict:
        """Fetch odometer data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.ODOMETER)
    
    async def _fetch_last_position(self, vehicle_name: str) -> Dict:
        """Fetch last position data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.LAST_POSITION)
    
    async def _fetch_trips(self, vehicle_name: str) -> Dict:
        """Fetch trips data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.TRIPS)
    
    async def _fetch_parking(self, vehicle_name: str) -> Dict:
        """Fetch parking data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.PARKING)
    
    async def _fetch_consumption(self, vehicle_name: str) -> Dict:
        """Fetch consumption data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.CONSUMPTION)
    
    async def _fetch_voltage(self, vehicle_name: str) -> Dict:
        """Fetch voltage data"""
        return await self.data_fetcher.fetch_data(vehicle_name, GPSDataType.VOLTAGE)
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _extract_speed(self, speed_data: Dict) -> Optional[float]:
        """Extract speed from speed data"""
        if not speed_data:
            return None
        speed_str = speed_data.get('speed', '0 km/h')
        return self._safe_float(speed_str.replace(' km/h', ''))
    
    def _extract_odometer(self, odometer_data: Dict) -> Optional[float]:
        """Extract odometer reading"""
        if not odometer_data:
            return None
        odo_str = odometer_data.get('odo', '0 km')
        return self._safe_float(odo_str.replace(' km', ''))
    
    def _extract_position(self, position_data: Dict) -> Optional[GPSPosition]:
        """Extract GPS position"""
        if not position_data or position_data.get('x') == 'checkDayBefore':
            return None
        return GPSPosition.from_api_response(position_data)
    
    def _extract_trips(self, trips_data: Dict) -> Optional[GPSTripData]:
        """Extract trip data"""
        if not trips_data or trips_data.get('totalKm') == '0.00 km':
            return GPSTripData(count=0, total_duration="0:00:00", total_km=0.0)
        return GPSTripData.from_api_response(trips_data)
    
    async def get_vehicle_list(self) -> List[str]:
        """Get list of available vehicles (mock implementation)"""
        # This would need to be implemented based on actual GPS IoT API
        # For now, return a sample list
        return ["1006", "1008", "1009", "1010", "1011", "1012", "1014", "1016", "1017", "1018", "1019"]
    
    async def close(self):
        """Close the GPS IoT client"""
        await self.data_fetcher.close()
        logger.info("GPS IoT Data Client closed")
