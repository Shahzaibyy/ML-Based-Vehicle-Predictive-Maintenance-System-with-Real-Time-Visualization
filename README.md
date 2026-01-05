# Production-Ready Vehicle Predictive Maintenance System

## Overview

This is a production-ready Machine Learning predictive maintenance service for vehicle fleets. It consumes real-time IoT sensor data from vehicles, runs daily predictions, and sends maintenance alerts to a MERN backend via REST API. The system is designed for ride-hailing vehicles (Uber/Careem type usage) sold on installments.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Vehicle IoT    │───▶│  ML Service      │───▶│  MERN Backend   │
│  Data API       │    │  (FastAPI)       │    │  (Node.js)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Daily Scheduler │
                       │  (Cron-style)    │
                       └──────────────────┘
```

## Features

- **Production-Ready ML Service**: Clean separation of training and inference
- **Real-Time Predictions**: FastAPI-based REST API for instant predictions
- **Daily Batch Processing**: Automated daily predictions for all vehicles
- **External API Integration**: Consumes vehicle IoT data via REST API
- **Mock Data Support**: Built-in dummy data generator for testing
- **Backend Integration**: Sends predictions to MERN backend via REST API
- **Stateless Design**: Scalable microservice architecture
- **Comprehensive Testing**: Frontend testing interface with dummy data

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Start the ML Service

```bash
# With mock data (for testing)
USE_MOCK_DATA=true python src/api/main.py

# With real vehicle API
USE_MOCK_DATA=false VEHICLE_API_URL=https://your-api.com python src/api/main.py
```

### 4. Test the Service

Open `frontend_test/test_client.html` in your browser to test the API.

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Data Source
USE_MOCK_DATA=true                    # Use mock data for testing
VEHICLE_API_URL=https://api.example.com  # Real vehicle API URL
VEHICLE_API_KEY=your_api_key_here     # API key for vehicle data

# Backend Integration
BACKEND_API_URL=http://localhost:3000 # MERN backend URL

# Scheduler
DAILY_PREDICTION_HOUR=2               # UTC hour for daily predictions (2 AM)
PREDICTION_BATCH_SIZE=10              # Vehicles per batch
```

## API Endpoints

### Core Prediction Endpoints

- `POST /predict` - Single vehicle prediction
- `POST /predict/batch` - Batch vehicle predictions
- `GET /health` - Health check and model status
- `GET /model/features` - Model feature importance

### Vehicle Data Endpoints

- `GET /vehicles` - List all vehicles
- `GET /vehicles/{vehicle_id}/latest-data` - Latest sensor data
- `POST /trigger-daily-predictions` - Manual daily prediction trigger

### Sample Request

```json
POST /predict
{
  "sensor_data": {
    "vehicle_id": "VEH-001",
    "Engine rpm": 800,
    "Lub oil pressure": 3.0,
    "Fuel pressure": 6.0,
    "Coolant pressure": 2.0,
    "lub oil temp": 77.0,
    "Coolant temp": 78.0,
    "Temperature_difference": 1.0
  }
}
```

### Sample Response

```json
{
  "vehicle_id": "VEH-001",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "maintenance_required": false,
  "maintenance_probability": 15.5,
  "estimated_days_remaining_before_maintenance": 45,
  "model_confidence": 84.5,
  "sensor_data": { ... }
}
```

## Integration with MERN Backend

### Backend Endpoint Requirements

The ML service sends predictions to your MERN backend at:

```
POST /api/vehicle/maintenance-prediction
```

### Payload Format

```json
{
  "vehicle_id": "VEH-001",
  "maintenance_required": false,
  "maintenance_probability": 15.5,
  "estimated_days_remaining_before_maintenance": 45,
  "prediction_timestamp": "2024-01-15T10:30:00.000Z",
  "sensor_data": { ... },
  "model_confidence": 84.5
}
```

### Backend Implementation Example

```javascript
// Express.js endpoint example
app.post('/api/vehicle/maintenance-prediction', async (req, res) => {
  try {
    const {
      vehicle_id,
      maintenance_required,
      maintenance_probability,
      estimated_days_remaining_before_maintenance,
      prediction_timestamp,
      sensor_data,
      model_confidence
    } = req.body;
    
    // Save to database
    await MaintenancePrediction.create({
      vehicle_id,
      maintenance_required,
      maintenance_probability,
      estimated_days_remaining_before_maintenance,
      prediction_timestamp,
      sensor_data,
      model_confidence
    });
    
    // Trigger notifications if maintenance required
    if (maintenance_required) {
      await notificationService.sendMaintenanceAlert(vehicle_id);
    }
    
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## Testing

### Frontend Testing Interface

1. Open `frontend_test/test_client.html` in your browser
2. Configure the API URL (default: `http://localhost:8000`)
3. Use the interface to:
   - Test single predictions
   - Test batch predictions
   - Generate random/anomaly data
   - View vehicle data
   - Trigger daily predictions

### Mock Data Testing

The system includes a comprehensive mock data generator that simulates:

- Realistic sensor readings with normal variations
- Anomalous data for testing edge cases
- Multiple vehicles with different profiles
- Historical data patterns

### API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "vehicle_id": "VEH-001",
      "Engine rpm": 800,
      "Lub oil pressure": 3.0,
      "Fuel pressure": 6.0,
      "Coolant pressure": 2.0,
      "lub oil temp": 77.0,
      "Coolant temp": 78.0,
      "Temperature_difference": 1.0
    }
  }'

# Get vehicles
curl http://localhost:8000/vehicles

# Trigger daily predictions
curl -X POST http://localhost:8000/trigger-daily-predictions
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Environment Variables**: Configure all settings via environment variables
2. **Model Storage**: Ensure persistent storage for the trained model
3. **Logging**: Configure structured logging for monitoring
4. **Health Checks**: Implement proper health monitoring
5. **Rate Limiting**: Add rate limiting for API endpoints
6. **Authentication**: Secure API endpoints if needed

## Monitoring and Logging

The service provides comprehensive logging:

- Model loading and prediction events
- API request/response logging
- Error tracking and debuggingl
- Daily prediction job status

Log levels can be configured via `LOG_LEVEL` environment variable.

## Machine Learning Model Analysis

### Model Architecture

**Algorithm**: Gradient Boosting Machine (GBM) Classifier  
**Library**: scikit-learn (GradientBoostingClassifier)  
**Type**: Supervised Binary Classification  

### Model Characteristics

#### ✅ **Dynamic Machine Learning Model**
This is a **fully dynamic ML model**, NOT static. Key characteristics:

- **Trained on real data**: Uses historical vehicle sensor data from `engine_data.csv`
- **Feature engineering**: Creates new features like `Temperature_difference`
- **Adaptive predictions**: Predictions vary based on real-time sensor inputs
- **Probabilistic outputs**: Provides confidence scores and probability estimates
- **Contextual estimates**: Calculates dynamic maintenance days based on sensor conditions

#### **Model Parameters**
```python
GradientBoostingClassifier(
    n_estimators=100,           # Number of boosting stages
    learning_rate=0.1,          # Step size shrinkage
    max_depth=3,                # Maximum tree depth
    random_state=42,            # Reproducibility
    max_features='sqrt',        # Feature selection strategy
    min_samples_leaf=5,         # Minimum samples per leaf
    min_samples_split=2,        # Minimum samples to split
    subsample=0.8               # Fraction of samples for each tree
)
```

### Feature Engineering

#### **Input Features (7 total)**
1. **Engine_rpm** - Engine revolutions per minute
2. **Lub_oil_pressure** - Lubricating oil pressure
3. **Fuel_pressure** - Fuel system pressure
4. **Coolant_pressure** - Engine coolant pressure
5. **lub_oil_temp** - Lubricating oil temperature
6. **Coolant_temp** - Engine coolant temperature
7. **Temperature_difference** - *Engineered feature*: Coolant_temp - lub_oil_temp

#### **Target Variable**
- **Engine Condition** (Binary): 0 = Normal, 1 = Maintenance Required

#### **Data Preprocessing**
- **Feature Scaling**: StandardScaler applied to all features
- **Missing Values**: Rows with missing data are dropped
- **Feature Engineering**: Temperature difference calculated dynamically
- **Column Renaming**: Spaces replaced with underscores for API compatibility

### Model Performance Metrics

#### **Classification Performance**
- **Accuracy**: ~67% on test data
- **Precision**: 
  - Maintenance Required: 69%
  - Normal Condition: 59%
- **Recall**: 
  - Maintenance Required: 85%
  - Normal Condition: 36%
- **F1-Score**: 
  - Maintenance Required: 76%
  - Normal Condition: 45%

#### **Feature Importance Ranking**
1. **Coolant_pressure**: 70% importance
2. **Temperature_difference**: 70% importance  
3. **Lub_oil_pressure**: 21% importance
4. **Coolant_temp**: 21% importance
5. **Fuel_pressure**: 14% importance
6. **lub_oil_temp**: 14% importance
7. **Engine_rpm**: 7% importance

### Dynamic Prediction Logic

#### **Maintenance Probability Calculation**
The model uses the trained GBM to calculate:
- **Binary prediction** (0/1) based on sensor patterns
- **Probability score** (0-100%) from predict_proba()
- **Model confidence** = max(probability, 1-probability)

#### **Dynamic Days Until Maintenance**
**NOT a static lookup** - calculated dynamically based on:

1. **Base interval**: 30 days
2. **Probability factor**:
   - >80%: 0.1x (urgent - ~3 days)
   - >60%: 0.3x (high priority - ~9 days)
   - >40%: 0.6x (medium - ~18 days)
   - ≤40%: 1.0x (normal - ~30 days)

3. **Critical sensor factors**:
   - **High RPM** (>2000): 0.5x multiplier
   - **Temperature difference** (>20°C): 0.6x multiplier
   - **Abnormal pressures** (<1.0 or >6.0): 0.7x multiplier

4. **Final calculation**: `days = base_days × probability_factor × critical_factors`
5. **Range**: Clamped between 1-90 days

### Model Training Process

#### **Data Pipeline**
1. **Load data** from CSV file
2. **Preprocess** with feature engineering
3. **Split data**: 70% train, 30% test (stratified)
4. **Scale features** using StandardScaler
5. **Train model** with GBM algorithm
6. **Evaluate performance** with classification metrics
7. **Save model** with scaler and metadata

#### **Model Persistence**
- **Format**: Pickle file (.pkl)
- **Contents**: Model + Scaler + Feature columns
- **Location**: `models/vehicle_maintenance_model.pkl`
- **Versioning**: Single file with all components

### Real-Time Inference

#### **Single Prediction Flow**
1. **Validate input** sensor data
2. **Scale features** using trained StandardScaler
3. **Predict** using GBM model
4. **Calculate probability** and confidence
5. **Estimate days** using dynamic logic
6. **Return structured response**

#### **Batch Processing**
- Processes multiple vehicles sequentially
- Error handling per vehicle
- Maintains response consistency

### Model Limitations & Considerations

#### **Current Limitations**
- **Training data size**: Limited by available dataset
- **Feature scope**: Only 7 sensor features
- **Temporal aspects**: No time-series analysis
- **Vehicle diversity**: Single model for all vehicle types

#### **Production Considerations**
- **Model retraining**: Should be scheduled periodically
- **Feature drift**: Monitor sensor data distribution changes
- **Performance monitoring**: Track accuracy over time
- **A/B testing**: Compare with baseline maintenance schedules

#### **Scalability**
- **Inference speed**: ~1ms per prediction
- **Memory usage**: ~10MB model size
- **Batch capacity**: No inherent limit
- **Concurrent requests**: Stateless design allows scaling

### Model Validation

#### **Input Validation**
- **Required features**: All 7 features must be present
- **Data types**: Numeric values only
- **Range checking**: Validates for NaN/Inf values
- **Format compliance**: Ensures API compatibility

#### **Output Validation**
- **Probability range**: 0-100%
- **Days range**: 1-90 days
- **Confidence calculation**: Max of class probabilities
- **Error handling**: Graceful fallbacks

### Future Enhancements

#### **Potential Improvements**
1. **Additional features**: Vehicle age, mileage, maintenance history
2. **Time-series analysis**: Sequential sensor patterns
3. **Ensemble models**: Combine multiple algorithms
4. **Deep learning**: Neural networks for complex patterns
5. **Transfer learning**: Adapt to different vehicle types

#### **Model Monitoring**
1. **Drift detection**: Feature distribution changes
2. **Performance tracking**: Accuracy over time
3. **Prediction auditing**: Review maintenance outcomes
4. **Feedback loops**: Incorporate actual maintenance results

## Troubleshooting

### Common Issues

1. **Model Not Found**: Run `python train_model.py` to train and save the model
2. **API Connection Errors**: Check `BACKEND_API_URL` configuration
3. **Mock Data Issues**: Set `USE_MOCK_DATA=true` for testing
4. **Scheduler Not Running**: Verify daily predictions are scheduled

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python src/api/main.py
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

## License

This project is licensed under the MIT License.

## Support

For questions and support, please contact the development team.
