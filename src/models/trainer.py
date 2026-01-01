import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class VehicleMaintenanceTrainer:
    def __init__(self, model_path: str = "models/vehicle_maintenance_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Engine_rpm', 'Lub_oil_pressure', 'Fuel_pressure', 'Coolant_pressure',
            'lub_oil_temp', 'Coolant_temp', 'Temperature_difference'
        ]
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load training data from CSV file."""
        try:
            data = pd.read_csv(csv_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the training data."""
        try:
            # Create temperature difference feature
            data['Temperature_difference'] = data['Coolant temp'] - data['lub oil temp']
            
            # Rename columns to match API field names
            data = data.rename(columns={
                'Engine rpm': 'Engine_rpm',
                'Lub oil pressure': 'Lub_oil_pressure', 
                'Fuel pressure': 'Fuel_pressure',
                'Coolant pressure': 'Coolant_pressure',
                'lub oil temp': 'lub_oil_temp',
                'Coolant temp': 'Coolant_temp'
            })
            
            # Ensure all required columns exist
            missing_cols = set(self.feature_columns + ['Engine Condition']) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Handle missing values
            data = data.dropna()
            
            logger.info(f"Preprocessed data shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the Gradient Boosting model."""
        try:
            # Prepare features and target
            X = data[self.feature_columns]
            y = data['Engine Condition']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                max_features='sqrt',
                min_samples_leaf=5,
                min_samples_split=2,
                subsample=0.8
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model and scaler
            self.save_model()
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
            }
            
            logger.info(f"Model trained with accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def save_model(self):
        """Save the trained model and scaler."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
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

if __name__ == "__main__":
    # Example usage
    trainer = VehicleMaintenanceTrainer()
    data = trainer.load_data("engine_data.csv")
    processed_data = trainer.preprocess_data(data)
    metrics = trainer.train_model(processed_data)
    print("Training completed successfully!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
