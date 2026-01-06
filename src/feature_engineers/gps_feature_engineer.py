"""
GPS Feature Engineering Module - Single Responsibility Principle
Extracts and engineers features from GPS IoT data for predictive maintenance
"""

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..data.gps_iot_client import GPSVehicleData, GPSPosition, GPSTripData

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature types for categorization - Single Responsibility"""
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    MAINTENANCE = "maintenance"
    PERFORMANCE = "performance"
    LOCATION = "location"

@dataclass
class EngineeredFeature:
    """Engineered feature with metadata - Single Responsibility"""
    name: str
    value: float
    feature_type: FeatureType
    description: str
    importance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input"""
        return {self.name: self.value}

class FeatureValidator:
    """Feature validation - Single Responsibility Principle"""
    
    @staticmethod
    def validate_feature_value(value: Any, feature_name: str) -> Optional[float]:
        """Validate and normalize feature value"""
        if value is None:
            return None
        
        try:
            # Convert to float
            float_value = float(value)
            
            # Check for invalid values
            if np.isnan(float_value) or np.isinf(float_value):
                logger.warning(f"Invalid value for {feature_name}: {value}")
                return None
            
            return float_value
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {feature_name} value to float: {value}")
            return None
    
    @staticmethod
    def clamp_value(value: float, min_val: float, max_val: float) -> float:
        """Clamp value to specified range"""
        return max(min_val, min(max_val, value))

class TemporalFeatureEngineer:
    """Temporal feature engineering - Single Responsibility Principle"""
    
    def __init__(self, validator: FeatureValidator):
        self.validator = validator
    
    def create_temporal_features(self, gps_data: GPSVehicleData) -> List[EngineeredFeature]:
        """Create temporal-based features from GPS data"""
        features = []
        
        # Hours since last movement
        hours_since_movement = self._calculate_hours_since_last_movement(gps_data)
        features.append(EngineeredFeature(
            name="hours_since_last_movement",
            value=hours_since_movement,
            feature_type=FeatureType.TEMPORAL,
            description="Hours since vehicle was last moved",
            importance_score=0.8
        ))
        
        # Daily usage hours (estimated from trips)
        daily_usage_hours = self._calculate_daily_usage_hours(gps_data)
        features.append(EngineeredFeature(
            name="daily_usage_hours",
            value=daily_usage_hours,
            feature_type=FeatureType.TEMPORAL,
            description="Estimated daily usage hours",
            importance_score=0.7
        ))
        
        # Time of day usage pattern
        time_of_day_score = self._calculate_time_of_day_pattern(gps_data)
        features.append(EngineeredFeature(
            name="time_of_day_usage_score",
            value=time_of_day_score,
            feature_type=FeatureType.TEMPORAL,
            description="Time of day usage pattern score",
            importance_score=0.6
        ))
        
        return features
    
    def _calculate_hours_since_last_movement(self, gps_data: GPSVehicleData) -> float:
        """Calculate hours since last vehicle movement"""
        if not gps_data.last_position:
            return 24.0  # Default to 24 hours if no position data
        
        now = datetime.utcnow()
        last_movement = gps_data.last_position.timestamp
        hours_diff = (now - last_movement).total_seconds() / 3600.0
        
        return self.validator.clamp_value(hours_diff, 0.0, 168.0)  # Max 1 week
    
    def _calculate_daily_usage_hours(self, gps_data: GPSVehicleData) -> float:
        """Estimate daily usage hours from trip data"""
        if not gps_data.trips:
            return 0.0
        
        # Parse total duration (format: "H:MM:SS")
        duration_str = gps_data.trips.total_duration
        if duration_str == "0:00:00":
            return 0.0
        
        try:
            parts = duration_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            total_hours = hours + minutes / 60.0
            
            return self.validator.clamp_value(total_hours, 0.0, 24.0)
        except (ValueError, IndexError):
            return 0.0
    
    def _calculate_time_of_day_pattern(self, gps_data: GPSVehicleData) -> float:
        """Calculate time of day usage pattern score"""
        if not gps_data.last_position:
            return 0.5  # Neutral score
        
        hour = gps_data.last_position.timestamp.hour
        
        # Business hours (6 AM - 8 PM) get higher scores
        if 6 <= hour <= 20:
            return 0.8
        else:
            return 0.3  # Night usage

class BehavioralFeatureEngineer:
    """Behavioral feature engineering - Single Responsibility Principle"""
    
    def __init__(self, validator: FeatureValidator):
        self.validator = validator
    
    def create_behavioral_features(self, gps_data: GPSVehicleData) -> List[EngineeredFeature]:
        """Create behavioral features from GPS data"""
        features = []
        
        # Usage intensity score
        usage_intensity = self._calculate_usage_intensity(gps_data)
        features.append(EngineeredFeature(
            name="usage_intensity_score",
            value=usage_intensity,
            feature_type=FeatureType.BEHAVIORAL,
            description="Vehicle usage intensity score",
            importance_score=0.9
        ))
        
        # Speed variance score
        speed_variance = self._calculate_speed_variance(gps_data)
        features.append(EngineeredFeature(
            name="speed_variance_score",
            value=speed_variance,
            feature_type=FeatureType.BEHAVIORAL,
            description="Speed variance indicator",
            importance_score=0.7
        ))
        
        # Trip frequency score
        trip_frequency = self._calculate_trip_frequency(gps_data)
        features.append(EngineeredFeature(
            name="trip_frequency_score",
            value=trip_frequency,
            feature_type=FeatureType.BEHAVIORAL,
            description="Daily trip frequency score",
            importance_score=0.8
        ))
        
        return features
    
    def _calculate_usage_intensity(self, gps_data: GPSVehicleData) -> float:
        """Calculate usage intensity based on multiple factors"""
        factors = []
        
        # Trip count factor
        if gps_data.trips:
            trip_count = gps_data.trips.count
            if trip_count == 0:
                factors.append(0.0)
            elif trip_count <= 5:
                factors.append(0.3)
            elif trip_count <= 10:
                factors.append(0.6)
            else:
                factors.append(1.0)
        else:
            factors.append(0.0)
        
        # Distance factor
        if gps_data.trips and gps_data.trips.total_km > 0:
            daily_km = gps_data.trips.total_km
            if daily_km <= 50:
                factors.append(0.3)
            elif daily_km <= 100:
                factors.append(0.6)
            else:
                factors.append(1.0)
        else:
            factors.append(0.0)
        
        # Engine status factor
        if gps_data.engine_status is not None:
            factors.append(float(gps_data.engine_status))
        else:
            factors.append(0.0)
        
        # Average the factors
        return np.mean(factors) if factors else 0.0
    
    def _calculate_speed_variance(self, gps_data: GPSVehicleData) -> float:
        """Calculate speed variance indicator"""
        if not gps_data.speed:
            return 0.0
        
        current_speed = gps_data.speed
        
        # Simple variance based on current speed vs typical ranges
        if current_speed == 0:
            return 0.1  # Low variance when stopped
        elif current_speed <= 30:
            return 0.3  # Low speed, low variance
        elif current_speed <= 60:
            return 0.6  # Moderate speed, moderate variance
        else:
            return 0.9  # High speed, high variance
    
    def _calculate_trip_frequency(self, gps_data: GPSVehicleData) -> float:
        """Calculate trip frequency score"""
        if not gps_data.trips:
            return 0.0
        
        trip_count = gps_data.trips.count
        
        # Normalize trip count (assuming max 20 trips per day)
        normalized_count = min(trip_count / 20.0, 1.0)
        return normalized_count

class MaintenanceFeatureEngineer:
    """Maintenance-specific feature engineering - Single Responsibility Principle"""
    
    def __init__(self, validator: FeatureValidator):
        self.validator = validator
    
    def create_maintenance_features(self, gps_data: GPSVehicleData) -> List[EngineeredFeature]:
        """Create maintenance-related features from GPS data"""
        features = []
        
        # Engine hours estimate
        engine_hours = self._estimate_engine_hours(gps_data)
        features.append(EngineeredFeature(
            name="engine_hours_estimate",
            value=engine_hours,
            feature_type=FeatureType.MAINTENANCE,
            description="Estimated engine operating hours",
            importance_score=0.9
        ))
        
        # Wear and tear score
        wear_score = self._calculate_wear_and_tear_score(gps_data)
        features.append(EngineeredFeature(
            name="wear_and_tear_score",
            value=wear_score,
            feature_type=FeatureType.MAINTENANCE,
            description="Vehicle wear and tear score",
            importance_score=0.8
        ))
        
        # Mileage since last service (mock implementation)
        mileage_since_service = self._calculate_mileage_since_service(gps_data)
        features.append(EngineeredFeature(
            name="mileage_since_last_service",
            value=mileage_since_service,
            feature_type=FeatureType.MAINTENANCE,
            description="Estimated mileage since last service",
            importance_score=0.9
        ))
        
        return features
    
    def _estimate_engine_hours(self, gps_data: GPSVehicleData) -> float:
        """Estimate engine operating hours"""
        if not gps_data.trips:
            return 0.0
        
        # Use trip duration as proxy for engine hours
        duration_str = gps_data.trips.total_duration
        if duration_str == "0:00:00":
            return 0.0
        
        try:
            parts = duration_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            total_hours = hours + minutes / 60.0
            
            return self.validator.clamp_value(total_hours, 0.0, 24.0)
        except (ValueError, IndexError):
            return 0.0
    
    def _calculate_wear_and_tear_score(self, gps_data: GPSVehicleData) -> float:
        """Calculate wear and tear score based on usage patterns"""
        factors = []
        
        # Odometer factor
        if gps_data.odometer_km:
            odometer = gps_data.odometer_km
            if odometer <= 50000:
                factors.append(0.2)  # Low wear
            elif odometer <= 100000:
                factors.append(0.5)  # Medium wear
            elif odometer <= 200000:
                factors.append(0.8)  # High wear
            else:
                factors.append(1.0)  # Very high wear
        else:
            factors.append(0.5)  # Default medium wear
        
        # Usage intensity factor
        if gps_data.trips:
            daily_km = gps_data.trips.total_km
            if daily_km <= 50:
                factors.append(0.3)
            elif daily_km <= 100:
                factors.append(0.6)
            else:
                factors.append(1.0)
        else:
            factors.append(0.3)
        
        return np.mean(factors) if factors else 0.5
    
    def _calculate_mileage_since_last_service(self, gps_data: GPSVehicleData) -> float:
        """Calculate estimated mileage since last service"""
        if not gps_data.odometer_km:
            return 0.0
        
        # Mock implementation - assume service every 10,000 km
        odometer = gps_data.odometer_km
        mileage_since_service = odometer % 10000
        
        return self.validator.clamp_value(mileage_since_service, 0.0, 10000.0)

class PerformanceFeatureEngineer:
    """Performance feature engineering - Single Responsibility Principle"""
    
    def __init__(self, validator: FeatureValidator):
        self.validator = validator
    
    def create_performance_features(self, gps_data: GPSVehicleData) -> List[EngineeredFeature]:
        """Create performance-related features from GPS data"""
        features = []
        
        # Current performance score
        performance_score = self._calculate_performance_score(gps_data)
        features.append(EngineeredFeature(
            name="current_performance_score",
            value=performance_score,
            feature_type=FeatureType.PERFORMANCE,
            description="Current vehicle performance score",
            importance_score=0.8
        ))
        
        # Fuel efficiency estimate
        fuel_efficiency = self._estimate_fuel_efficiency(gps_data)
        features.append(EngineeredFeature(
            name="fuel_efficiency_estimate",
            value=fuel_efficiency,
            feature_type=FeatureType.PERFORMANCE,
            description="Estimated fuel efficiency (km/l)",
            importance_score=0.6
        ))
        
        return features
    
    def _calculate_performance_score(self, gps_data: GPSVehicleData) -> float:
        """Calculate overall performance score"""
        factors = []
        
        # Engine status factor
        if gps_data.engine_status is not None:
            factors.append(1.0 if gps_data.engine_status == 1 else 0.0)
        else:
            factors.append(0.5)
        
        # Speed factor (normal operation)
        if gps_data.speed is not None:
            if gps_data.speed == 0:
                factors.append(0.5)  # Neutral when stopped
            elif 20 <= gps_data.speed <= 80:
                factors.append(1.0)  # Good operating range
            else:
                factors.append(0.3)  # Outside optimal range
        else:
            factors.append(0.5)
        
        # Voltage factor (if available)
        if gps_data.voltage is not None:
            if 12.0 <= gps_data.voltage <= 14.5:
                factors.append(1.0)  # Good voltage
            else:
                factors.append(0.3)  # Poor voltage
        else:
            factors.append(0.5)
        
        return np.mean(factors) if factors else 0.5
    
    def _estimate_fuel_efficiency(self, gps_data: GPSVehicleData) -> float:
        """Estimate fuel efficiency from available data"""
        if not gps_data.trips or not gps_data.trips.total_km:
            return 0.0
        
        # Mock implementation - typical values for ride-hailing vehicles
        daily_km = gps_data.trips.total_km
        
        if daily_km <= 50:
            return 12.0  # km/l - better efficiency for short trips
        elif daily_km <= 100:
            return 10.0  # km/l - moderate efficiency
        else:
            return 8.0   # km/l - lower efficiency for heavy usage

class GPSFeatureEngineer:
    """Main GPS feature engineer - Facade Pattern for simplified interface"""
    
    def __init__(self):
        self.validator = FeatureValidator()
        
        # Dependency Injection - SOLID Principle
        self.temporal_engineer = TemporalFeatureEngineer(self.validator)
        self.behavioral_engineer = BehavioralFeatureEngineer(self.validator)
        self.maintenance_engineer = MaintenanceFeatureEngineer(self.validator)
        self.performance_engineer = PerformanceFeatureEngineer(self.validator)
        
        logger.info("GPS Feature Engineer initialized")
    
    def create_all_features(self, gps_data: GPSVehicleData) -> Dict[str, float]:
        """Create all engineered features from GPS data"""
        all_features = {}
        
        try:
            # Create features from different engineers
            temporal_features = self.temporal_engineer.create_temporal_features(gps_data)
            behavioral_features = self.behavioral_engineer.create_behavioral_features(gps_data)
            maintenance_features = self.maintenance_engineer.create_maintenance_features(gps_data)
            performance_features = self.performance_engineer.create_performance_features(gps_data)
            
            # Combine all features
            all_engineered_features = (
                temporal_features + 
                behavioral_features + 
                maintenance_features + 
                performance_features
            )
            
            # Convert to dictionary and validate
            for feature in all_engineered_features:
                validated_value = self.validator.validate_feature_value(
                    feature.value, feature.name
                )
                if validated_value is not None:
                    all_features[feature.name] = validated_value
                else:
                    logger.warning(f"Invalid value for feature {feature.name}: {feature.value}")
            
            logger.info(f"Created {len(all_features)} engineered features for {gps_data.vehicle_name}")
            return all_features
            
        except Exception as e:
            logger.error(f"Error creating features for {gps_data.vehicle_name}: {e}")
            return {}
    
    def create_sensor_compatible_features(self, gps_data: GPSVehicleData) -> Dict[str, float]:
        """Create features compatible with existing sensor model"""
        # Map GPS features to sensor feature names for backward compatibility
        gps_features = self.create_all_features(gps_data)
        
        # Create sensor-compatible feature mapping
        sensor_features = {
            # Map GPS features to existing sensor feature names
            'Engine_rpm': self._map_to_engine_rpm(gps_features),
            'Lub_oil_pressure': self._map_to_lub_oil_pressure(gps_features),
            'Fuel_pressure': self._map_to_fuel_pressure(gps_features),
            'Coolant_pressure': self._map_to_coolant_pressure(gps_features),
            'lub_oil_temp': self._map_to_lub_oil_temp(gps_features),
            'Coolant_temp': self._map_to_coolant_temp(gps_features),
            'Temperature_difference': self._map_to_temperature_difference(gps_features)
        }
        
        return sensor_features
    
    def _map_to_engine_rpm(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to engine RPM equivalent"""
        # Use usage intensity and speed to estimate RPM
        usage_intensity = gps_features.get('usage_intensity_score', 0.5)
        speed_variance = gps_features.get('speed_variance_score', 0.5)
        
        # Estimate RPM based on usage patterns
        estimated_rpm = 800 + (usage_intensity * 1200) + (speed_variance * 600)
        return min(estimated_rpm, 3000)  # Cap at reasonable max
    
    def _map_to_lub_oil_pressure(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to lubricating oil pressure equivalent"""
        wear_score = gps_features.get('wear_and_tear_score', 0.5)
        performance_score = gps_features.get('current_performance_score', 0.5)
        
        # Estimate pressure based on wear and performance
        estimated_pressure = 4.0 - (wear_score * 1.5) + (performance_score * 0.5)
        return max(estimated_pressure, 1.0)  # Minimum pressure
    
    def _map_to_fuel_pressure(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to fuel pressure equivalent"""
        performance_score = gps_features.get('current_performance_score', 0.5)
        
        # Estimate fuel pressure based on performance
        estimated_pressure = 5.0 + (performance_score * 3.0)
        return min(estimated_pressure, 10.0)  # Cap at reasonable max
    
    def _map_to_coolant_pressure(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to coolant pressure equivalent"""
        wear_score = gps_features.get('wear_and_tear_score', 0.5)
        
        # Estimate coolant pressure based on wear
        estimated_pressure = 2.0 + (wear_score * 1.5)
        return min(estimated_pressure, 4.0)  # Cap at reasonable max
    
    def _map_to_lub_oil_temp(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to lubricating oil temperature equivalent"""
        usage_intensity = gps_features.get('usage_intensity_score', 0.5)
        daily_usage = gps_features.get('daily_usage_hours', 0.0)
        
        # Estimate temperature based on usage
        estimated_temp = 75 + (usage_intensity * 10) + (daily_usage * 0.5)
        return min(estimated_temp, 95.0)  # Cap at reasonable max
    
    def _map_to_coolant_temp(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to coolant temperature equivalent"""
        lub_oil_temp = self._map_to_lub_oil_temp(gps_features)
        
        # Coolant temp typically slightly higher than oil temp
        estimated_temp = lub_oil_temp + 2.0
        return min(estimated_temp, 100.0)  # Cap at reasonable max
    
    def _map_to_temperature_difference(self, gps_features: Dict[str, float]) -> float:
        """Map GPS features to temperature difference equivalent"""
        coolant_temp = self._map_to_coolant_temp(gps_features)
        lub_oil_temp = self._map_to_lub_oil_temp(gps_features)
        
        return coolant_temp - lub_oil_temp
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {
            'usage_intensity_score': 0.9,
            'engine_hours_estimate': 0.9,
            'mileage_since_last_service': 0.9,
            'hours_since_last_movement': 0.8,
            'trip_frequency_score': 0.8,
            'wear_and_tear_score': 0.8,
            'current_performance_score': 0.8,
            'daily_usage_hours': 0.7,
            'speed_variance_score': 0.7,
            'time_of_day_usage_score': 0.6,
            'fuel_efficiency_estimate': 0.6
        }
