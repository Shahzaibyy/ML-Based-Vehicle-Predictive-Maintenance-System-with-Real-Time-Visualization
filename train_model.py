#!/usr/bin/env python3
"""
Training script for Vehicle Predictive Maintenance Model
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.trainer import VehicleMaintenanceTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        logger.info("Starting vehicle maintenance model training")
        
        # Initialize trainer
        trainer = VehicleMaintenanceTrainer()
        
        # Load and preprocess data
        data_path = "engine_data.csv"
        if not Path(data_path).exists():
            logger.error(f"Training data not found: {data_path}")
            sys.exit(1)
        
        logger.info(f"Loading training data from {data_path}")
        data = trainer.load_data(data_path)
        
        logger.info("Preprocessing data")
        processed_data = trainer.preprocess_data(data)
        
        # Train model
        logger.info("Training model")
        metrics = trainer.train_model(processed_data)
        
        # Display results
        logger.info("Training completed successfully!")
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Class 0): {metrics['classification_report']['0']['precision']:.4f}")
        print(f"Recall (Class 0): {metrics['classification_report']['0']['recall']:.4f}")
        print(f"F1-Score (Class 0): {metrics['classification_report']['0']['f1-score']:.4f}")
        print(f"Precision (Class 1): {metrics['classification_report']['1']['precision']:.4f}")
        print(f"Recall (Class 1): {metrics['classification_report']['1']['recall']:.4f}")
        print(f"F1-Score (Class 1): {metrics['classification_report']['1']['f1-score']:.4f}")
        
        print("\nFeature Importance:")
        for feature, importance in sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
        
        print(f"\nModel saved to: {trainer.model_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
