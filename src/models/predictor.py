import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum

from ..data.gps_iot_client import GPSVehicleData
from ..feature_engineers.gps_feature_engineer import GPSFeatureEngineer

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Data source types - Single Responsibility Principle"""
    SENSOR = "sensor"
    GPS_IOT = "gps_iot"
    HYBRID = "hybrid"

class DataProcessor(ABC):
    """Abstract data processor - Dependency Inversion Principle"""
    
    @abstractmethod
    def process(self, data: Union[Dict, GPSVehicleData]) -> Dict[str, float]:
        """Process data and return features"""
        pass

class SensorDataProcessor(DataProcessor):
    """Sensor data processor - Liskov Substitution Principle"""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
    
    def process(self, data: Dict) -> Dict[str, float]:
        """Process sensor data"""
        # Validate and extract sensor features
        features = {}
        for feature in self.feature_columns:
            if feature in data:
                value = data[feature]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    features[feature] = float(value)
                else:
                    logger.warning(f"Invalid sensor value for {feature}: {value}")
                    features[feature] = 0.0  # Default value
            else:
                logger.warning(f"Missing sensor feature: {feature}")
                features[feature] = 0.0  # Default value
        
        return features

class GPSDataProcessor(DataProcessor):
    """GPS IoT data processor - Liskov Substitution Principle"""
    
    def __init__(self):
        self.feature_engineer = GPSFeatureEngineer()
    
    def process(self, data: GPSVehicleData) -> Dict[str, float]:
        """Process GPS IoT data"""
        if not isinstance(data, GPSVehicleData):
            raise ValueError("GPSDataProcessor expects GPSVehicleData")
        
        # Create sensor-compatible features from GPS data
        return self.feature_engineer.create_sensor_compatible_features(data)

class HybridDataProcessor(DataProcessor):
    """Hybrid data processor for sensor + GPS data - Liskov Substitution Principle"""
    
    def __init__(self, feature_columns: List[str]):
        self.sensor_processor = SensorDataProcessor(feature_columns)
        self.gps_processor = GPSDataProcessor()
        self.feature_engineer = GPSFeatureEngineer()
    
    def process(self, data: Dict) -> Dict[str, float]:
        """Process hybrid sensor + GPS data"""
        sensor_data = data.get('sensor_data', {})
        gps_data = data.get('gps_data')
        
        # Process sensor data
        sensor_features = self.sensor_processor.process(sensor_data)
        
        # Process GPS data if available
        if gps_data:
            gps_features = self.gps_processor.process(gps_data)
            
            # Enhance sensor features with GPS insights
            enhanced_features = self._enhance_sensor_features_with_gps(
                sensor_features, gps_features, gps_data
            )
            return enhanced_features
        
        return sensor_features
    
    def _enhance_sensor_features_with_gps(
        self, 
        sensor_features: Dict[str, float], 
        gps_features: Dict[str, float], 
        gps_data: GPSVehicleData
    ) -> Dict[str, float]:
        """Enhance sensor features with GPS insights"""
        enhanced = sensor_features.copy()
        
        # Use GPS data to adjust sensor readings
        if gps_data.speed is not None and gps_data.speed > 0:
            # Adjust RPM based on actual speed
            speed_factor = min(gps_data.speed / 60.0, 1.5)  # Normalize speed
            enhanced['Engine_rpm'] = enhanced['Engine_rpm'] * (0.8 + 0.4 * speed_factor)
        
        if gps_data.engine_status == 1:
            # Engine is on, adjust pressures and temperatures
            enhanced['Coolant_pressure'] *= 1.1
            enhanced['Coolant_temp'] *= 1.05
        
        # Add GPS-derived adjustments
        wear_score = self.feature_engineer.create_all_features(gps_data).get('wear_and_tear_score', 0.5)
        
        # Adjust pressures based on wear
        if wear_score > 0.7:
            enhanced['Lub_oil_pressure'] *= 0.9
            enhanced['Fuel_pressure'] *= 0.85
            enhanced['Coolant_pressure'] *= 0.9
        
        return enhanced

class VehicleMaintenancePredictor:
    """Enhanced Vehicle Maintenance Predictor with GPS IoT support - SOLID Principles"""
    
    def __init__(self, model_path: str = "models/vehicle_maintenance_model.pkl", data_source_type: DataSourceType = DataSourceType.SENSOR):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.data_source_type = data_source_type
        
        # Dependency Injection - SOLID Principle
        self.data_processor = self._create_data_processor(data_source_type)
        
        self.load_model()
    
    def _create_data_processor(self, data_source_type: DataSourceType) -> DataProcessor:
        """Factory method to create appropriate data processor - Factory Pattern"""
        if data_source_type == DataSourceType.SENSOR:
            return SensorDataProcessor(self.feature_columns)
        elif data_source_type == DataSourceType.GPS_IOT:
            return GPSDataProcessor()
        elif data_source_type == DataSourceType.HYBRID:
            return HybridDataProcessor(self.feature_columns)
        else:
            raise ValueError(f"Unsupported data source type: {data_source_type}")
    
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            # Recreate data processor with loaded feature columns
            self.data_processor = self._create_data_processor(self.data_source_type)
            
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, data: Union[Dict, GPSVehicleData]) -> Dict[str, Any]:
        """Predict maintenance for a single vehicle - supports multiple data sources."""
        try:
            # Process data based on source type
            features = self.data_processor.process(data)
            
            # Validate processed features
            missing_features = set(self.feature_columns) - set(features.keys())
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    features[feature] = 0.0
            
            # Prepare input for model
            input_data = np.array([features[feature] for feature in self.feature_columns]).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0, 1]
            
            # Calculate estimated days until maintenance
            estimated_days = self._calculate_estimated_days(features, probability, data)
            
            # Create result
            result = {
                'vehicle_id': self._extract_vehicle_id(data),
                'timestamp': datetime.utcnow().isoformat(),
                'maintenance_required': bool(prediction == 1),
                'maintenance_probability': round(probability * 100, 2),
                'estimated_days_remaining_before_maintenance': estimated_days,
                'model_confidence': round(max(probability, 1 - probability) * 100, 2),
                'data_source': self.data_source_type.value,
                'features_used': list(features.keys())
            }
            
            # Add GPS-specific information if available
            if isinstance(data, GPSVehicleData) or (isinstance(data, dict) and 'gps_data' in data):
                result['gps_insights'] = self._extract_gps_insights(data)
            
            logger.info(f"Prediction completed for vehicle {result['vehicle_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def predict_from_gps_iot(self, gps_data: GPSVehicleData) -> Dict[str, Any]:
        """Convenience method for GPS IoT data prediction - Interface Segregation Principle"""
        if self.data_source_type != DataSourceType.GPS_IOT:
            # Create GPS-specific predictor
            gps_predictor = VehicleMaintenancePredictor(
                self.model_path, 
                DataSourceType.GPS_IOT
            )
            return gps_predictor.predict_single(gps_data)
        
        return self.predict_single(gps_data)
    
    def predict_from_sensor(self, sensor_data: Dict) -> Dict[str, Any]:
        """Convenience method for sensor data prediction - Interface Segregation Principle"""
        if self.data_source_type != DataSourceType.SENSOR:
            # Create sensor-specific predictor
            sensor_predictor = VehicleMaintenancePredictor(
                self.model_path, 
                DataSourceType.SENSOR
            )
            return sensor_predictor.predict_single(sensor_data)
        
        return self.predict_single(sensor_data)
    
    def predict_hybrid(self, sensor_data: Dict, gps_data: GPSVehicleData) -> Dict[str, Any]:
        """Convenience method for hybrid data prediction - Interface Segregation Principle"""
        hybrid_data = {
            'sensor_data': sensor_data,
            'gps_data': gps_data
        }
        
        if self.data_source_type != DataSourceType.HYBRID:
            # Create hybrid-specific predictor
            hybrid_predictor = VehicleMaintenancePredictor(
                self.model_path, 
                DataSourceType.HYBRID
            )
            return hybrid_predictor.predict_single(hybrid_data)
        
        return self.predict_single(hybrid_data)
    
    def predict_batch(self, data_list: List[Union[Dict, GPSVehicleData]]) -> List[Dict[str, Any]]:
        """Predict maintenance for multiple vehicles - Open/Closed Principle"""
        results = []
        for data in data_list:
            try:
                result = self.predict_single(data)
                results.append(result)
            except Exception as e:
                vehicle_id = self._extract_vehicle_id(data)
                logger.error(f"Error predicting for vehicle {vehicle_id}: {e}")
                # Add error result
                results.append({
                    'vehicle_id': vehicle_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e),
                    'maintenance_required': False,
                    'maintenance_probability': 0.0,
                    'estimated_days_remaining_before_maintenance': 0,
                    'data_source': self.data_source_type.value
                })
        
        return results
    
    def _extract_vehicle_id(self, data: Union[Dict, GPSVehicleData]) -> str:
        """Extract vehicle ID from different data types - Single Responsibility"""
        if isinstance(data, GPSVehicleData):
            return data.vehicle_name
        elif isinstance(data, dict):
            # Try different possible keys
            return data.get('vehicle_id') or data.get('vehicle_name') or 'unknown'
        else:
            return 'unknown'
    
    def _extract_gps_insights(self, data: Union[Dict, GPSVehicleData]) -> Dict[str, Any]:
        """Extract GPS-specific insights for result - Single Responsibility"""
        insights = {}
        
        if isinstance(data, GPSVehicleData):
            insights = {
                'vin': data.vin,
                'engine_status': data.engine_status,
                'current_speed': data.speed,
                'odometer_km': data.odometer_km,
                'last_position': {
                    'latitude': data.last_position.latitude,
                    'longitude': data.last_position.longitude,
                    'timestamp': data.last_position.timestamp.isoformat()
                } if data.last_position else None,
                'daily_trips': data.trips.count if data.trips else 0,
                'daily_distance_km': data.trips.total_km if data.trips else 0
            }
        elif isinstance(data, dict) and 'gps_data' in data:
            gps_data = data['gps_data']
            if isinstance(gps_data, GPSVehicleData):
                insights = self._extract_gps_insights(gps_data)
        
        return insights
    
    def _calculate_estimated_days(
        self, 
        features: Dict[str, float], 
        probability: float, 
        data: Union[Dict, GPSVehicleData]
    ) -> int:
        """Calculate estimated days until maintenance - Enhanced with GPS data"""
        try:
            # Base calculation on probability
            base_days = 30
            
            # Probability factor
            if probability > 0.8:
                probability_factor = 0.1  # Very high probability - urgent
            elif probability > 0.6:
                probability_factor = 0.3  # High priority
            elif probability > 0.4:
                probability_factor = 0.6  # Medium priority
            else:
                probability_factor = 1.0  # Low priority
            
            # Critical sensor factors
            critical_factors = self._get_critical_factors(features)
            
            # GPS-specific factors
            gps_factors = self._get_gps_factors(data)
            
            # Calculate final days
            estimated_days = int(base_days * probability_factor * critical_factors * gps_factors)
            return max(1, min(estimated_days, 90))  # Clamp between 1 and 90 days
            
        except Exception as e:
            logger.error(f"Error calculating estimated days: {e}")
            return 30  # Default to 30 days on error
    
    def _get_gps_factors(self, data: Union[Dict, GPSVehicleData]) -> float:
        """Calculate GPS-specific factors - Single Responsibility"""
        factors = []
        
        if isinstance(data, GPSVehicleData):
            # Odometer factor
            if data.odometer_km:
                if data.odometer_km > 200000:
                    factors.append(0.5)  # High mileage - reduce days
                elif data.odometer_km > 100000:
                    factors.append(0.7)  # Medium mileage
                else:
                    factors.append(1.0)  # Low mileage
            
            # Usage intensity factor
            if data.trips and data.trips.count > 10:
                factors.append(0.8)  # Heavy usage
            elif data.trips and data.trips.count > 5:
                factors.append(0.9)  # Moderate usage
            else:
                factors.append(1.0)  # Light usage
            
            # Engine status factor
            if data.engine_status == 1:
                factors.append(0.9)  # Engine currently running
            else:
                factors.append(1.0)  # Engine off
        
        elif isinstance(data, dict) and 'gps_data' in data:
            gps_data = data['gps_data']
            if isinstance(gps_data, GPSVehicleData):
                return self._get_gps_factors(gps_data)
        
        return np.mean(factors) if factors else 1.0
    
    def _get_critical_factors(self, sensor_data: Dict[str, float]) -> float:
        """Calculate critical factors based on sensor readings."""
        factors = []
        
        # Engine RPM factor
        rpm = sensor_data.get('Engine_rpm', 0)
        if rpm > 2000:
            factors.append(0.5)  # High RPM - reduce days
        elif rpm > 1500:
            factors.append(0.7)
        else:
            factors.append(1.0)
        
        # Temperature difference factor
        temp_diff = sensor_data.get('Temperature_difference', 0)
        if abs(temp_diff) > 20:
            factors.append(0.6)  # Large temperature difference - reduce days
        elif abs(temp_diff) > 10:
            factors.append(0.8)
        else:
            factors.append(1.0)
        
        # Pressure factors
        pressures = ['Lub_oil_pressure', 'Fuel_pressure', 'Coolant_pressure']
        for pressure in pressures:
            value = sensor_data.get(pressure, 0)
            if value < 1.0 or value > 6.0:
                factors.append(0.7)  # Abnormal pressure - reduce days
            else:
                factors.append(1.0)
        
        return np.mean(factors)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return dict(zip(self.feature_columns, self.model.feature_importances_))
    
    def validate_sensor_data(self, sensor_data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate sensor data format and values."""
        errors = []
        
        # Check required features
        missing_features = set(self.feature_columns) - set(sensor_data.keys())
        if missing_features:
            errors.append(f"Missing features: {missing_features}")
        
        # Check data types and ranges
        for feature in self.feature_columns:
            if feature in sensor_data:
                value = sensor_data[feature]
                if not isinstance(value, (int, float)):
                    errors.append(f"Invalid type for {feature}: expected number, got {type(value)}")
                elif np.isnan(value) or np.isinf(value):
                    errors.append(f"Invalid value for {feature}: {value}")
        
        return len(errors) == 0, errors
