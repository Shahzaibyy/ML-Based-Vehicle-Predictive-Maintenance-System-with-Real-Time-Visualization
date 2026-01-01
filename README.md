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
- Error tracking and debugging
- Daily prediction job status

Log levels can be configured via `LOG_LEVEL` environment variable.

## Model Performance

The Gradient Boosting Machine model achieves:

- **Accuracy**: ~67% on test data
- **Precision**: 69% (maintenance required), 59% (normal)
- **Recall**: 85% (maintenance required), 36% (normal)
- **F1-Score**: 76% (maintenance required), 45% (normal)

### Feature Importance

1. Coolant pressure: 70% importance
2. Temperature_difference: 70% importance  
3. Lub oil pressure: 21% importance
4. Coolant temp: 21% importance
5. Fuel pressure: 14% importance
6. lub oil temp: 14% importance
7. Engine rpm: 7% importance

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
