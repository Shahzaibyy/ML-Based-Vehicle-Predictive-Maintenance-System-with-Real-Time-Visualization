import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VehicleMaintenancePredictor:
    def __init__(self, model_path: str = "models/vehicle_maintenance_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.load_model()
    
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
            
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict maintenance for a single vehicle."""
        try:
            # Validate input data
            missing_features = set(self.feature_columns) - set(sensor_data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Prepare input
            input_data = np.array([sensor_data[feature] for feature in self.feature_columns]).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0, 1]
            
            # Calculate estimated days until maintenance
            estimated_days = self._calculate_estimated_days(sensor_data, probability)
            
            result = {
                'vehicle_id': sensor_data.get('vehicle_id', 'unknown'),
                'timestamp': datetime.utcnow().isoformat(),
                'maintenance_required': bool(prediction == 1),
                'maintenance_probability': round(probability * 100, 2),
                'estimated_days_remaining_before_maintenance': estimated_days,
                'sensor_data': sensor_data,
                'model_confidence': round(max(probability, 1 - probability) * 100, 2)
            }
            
            logger.info(f"Prediction completed for vehicle {result['vehicle_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def predict_batch(self, sensor_data_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Predict maintenance for multiple vehicles."""
        results = []
        for sensor_data in sensor_data_list:
            try:
                result = self.predict_single(sensor_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for vehicle {sensor_data.get('vehicle_id', 'unknown')}: {e}")
                # Add error result
                results.append({
                    'vehicle_id': sensor_data.get('vehicle_id', 'unknown'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e),
                    'maintenance_required': False,
                    'maintenance_probability': 0.0,
                    'estimated_days_remaining_before_maintenance': 0
                })
        
        return results
    
    def _calculate_estimated_days(self, sensor_data: Dict[str, float], probability: float) -> int:
        """Calculate estimated days until maintenance based on sensor data and probability."""
        try:
            # Base calculation on probability and critical sensor readings
            base_days = 30  # Base maintenance interval
            
            # Adjust based on probability
            if probability > 0.8:
                probability_factor = 0.1  # Very high probability - maintenance needed soon
            elif probability > 0.6:
                probability_factor = 0.3  # High probability
            elif probability > 0.4:
                probability_factor = 0.6  # Medium probability
            else:
                probability_factor = 1.0  # Low probability
            
            # Adjust based on critical sensor readings
            critical_factors = self._get_critical_factors(sensor_data)
            
            # Calculate final days
            estimated_days = int(base_days * probability_factor * critical_factors)
            return max(1, min(estimated_days, 90))  # Clamp between 1 and 90 days
            
        except Exception as e:
            logger.error(f"Error calculating estimated days: {e}")
            return 30  # Default to 30 days on error
    
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
