# Crop Yield Prediction

A Flask-based API system for intelligent crop recommendation and yield prediction in hydroponic farming environments using machine learning and IoT sensor data.

## What This Does

- Collects and processes real-time sensor data (temperature, humidity, pH, rainfall) from IoT devices
- Provides ML-powered crop recommendations based on current environmental conditions
- Predicts crop yields with confidence scoring and prediction intervals
- Suggests actionable improvements to optimize growing conditions for specific crops
- Analyzes historical environmental trends and sensor data patterns
- Supports Arduino and ESP8266 hardware for sensor data collection and transmission

## Architecture

```
┌─────────────────┐
│   IoT Devices   │
│ (Arduino/ESP8266)│
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│   Flask API     │
│  (Controllers)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │        │
    ▼        ▼
┌────────┐ ┌────────┐
│  Data  │ │   ML   │
│Service │ │Service │
└───┬────┘ └───┬────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│    MongoDB      │
│  (Time-series)  │
└─────────────────┘
```

## Key Features

- RESTful API with API key authentication
- Random Forest models for crop classification and yield regression
- Rule-based fallback predictions when ML models are unavailable
- Data quality scoring and VPD (Vapor Pressure Deficit) calculations
- Multi-device support with device management and status tracking
- Historical data retrieval with configurable time-range filtering

## Tech Stack

**Backend**: Flask 2.2.3, Flask-CORS 3.0.10, PyMongo 4.3.3, Python 3.8+  
**Machine Learning**: scikit-learn 1.2.2, pandas 1.5.3, numpy 1.24.2, joblib 1.2.0  
**Database**: MongoDB 4.4+  
**Hardware**: Arduino (C++), ESP8266 (C++)  
**Other**: python-dotenv 1.0.0, gunicorn 20.1.0

## Repository Structure

```
crop_yield_prediction/
├── backend/
│   ├── app.py                    # Flask application entry point
│   ├── config.py                 # Configuration management
│   ├── requirements.txt          # Python dependencies
│   ├── controllers/              # API route handlers
│   ├── services/                 # Business logic
│   ├── models/                   # Data models
│   └── utils/                    # Helper functions
├── ml/
│   ├── train_model.py            # Crop prediction model training
│   ├── crop_yield_predictor.py   # Yield prediction model
│   ├── preprocess.py             # Data preprocessing
│   ├── evaluate.py               # Model evaluation
│   └── models/                   # Trained model files
├── hardware/
│   ├── arduino_code.ino          # Arduino sensor code
│   └── esp8266_code.ino          # ESP8266 WiFi module code
├── data/
│   └── cpdata.csv                # Crop dataset
└── LICENSE                        # MIT License
```

## Quickstart

### Prerequisites

- Python 3.8 or higher
- MongoDB 4.4 or higher (running locally or accessible)
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crop_yield_prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the `backend/` directory:
   ```env
   FLASK_ENV=development
   DB_URI=mongodb://localhost:27017/
   DB_NAME=hydroponic_system
   API_KEY=your-api-key-here
   SECRET_KEY=your-secret-key-here
   MODEL_PATH=ml/models/random_forest_model.pkl
   CROP_DATA_PATH=data/cpdata.csv
   DEBUG=True
   ```

5. **Train the ML models** (optional, pre-trained model included)
   ```bash
   # From project root
   python ml/train_model.py data/cpdata.csv
   python ml/crop_yield_predictor.py data/cpdata.csv ml/models/yield_prediction_model.pkl
   ```

6. **Start MongoDB** (if running locally)
   ```bash
   # On Windows (if installed as service):
   net start MongoDB
   # On Linux/Mac:
   mongod
   ```

### Run

1. **Start the Flask application**
   ```bash
   cd backend
   python app.py
   ```

   The API will be available at `http://localhost:5000`

2. **Verify the API is running**
   ```bash
   curl http://localhost:5000/health
   ```

## Configuration

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment (development/production/testing) | `development` |
| `DB_URI` | MongoDB connection URI | `mongodb://localhost:27017/` |
| `DB_NAME` | MongoDB database name | `hydroponic_system` |
| `API_KEY` | API key for authentication | `your-api-key-here` |
| `SECRET_KEY` | Flask secret key | `your-secret-key-here` |
| `MODEL_PATH` | Path to trained crop prediction model | `ml/models/random_forest_model.pkl` |
| `CROP_DATA_PATH` | Path to crop dataset CSV | `data/cpdata.csv` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `PORT` | Server port | `5000` |
| `ENABLE_NOTIFICATIONS` | Enable email notifications | `False` |
| `NOTIFICATION_EMAIL` | Email for notifications | `` |
| `SMTP_SERVER` | SMTP server address | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |

## Usage

### API Endpoints

All endpoints require an `X-API-Key` header with your API key (except `/` and `/health`).

**Sensor Data**
- `POST /api/sensor-data` - Submit sensor readings
- `GET /api/sensor-data/<device_id>` - Get sensor history (`?days=7&limit=100`)
- `GET /api/devices` - List all registered devices

**Predictions**
- `POST /api/predict` - Predict suitable crops for current conditions
- `POST /api/predict-yield` - Estimate crop yield
- `GET /api/optimal-conditions/<crop>` - Get optimal growing conditions
- `POST /api/suggest-improvements` - Get suggestions to improve conditions
- `GET /api/analyze-crop/<crop_name>` - Analyze historical data (`?days=30`)
- `GET /api/crops` - List all supported crops

### Example API Calls

```bash
# Health check
curl http://localhost:5000/health

# Submit sensor data
curl -X POST http://localhost:5000/api/sensor-data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"device_id": "sensor01", "temperature": 25.5, "humidity": 60, "ph": 6.8, "rainfall": 100}'

# Predict crop
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 150}'

# Predict yield
curl -X POST http://localhost:5000/api/predict-yield \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"crop": "rice", "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 150}'
```

### IoT Device Setup

1. **Arduino**: Upload `hardware/arduino_code.ino`, connect DS18B20 (temperature), pH sensor, humidity sensor
2. **ESP8266**: Upload `hardware/esp8266_code.ino`, update WiFi credentials (`WIFI_SSID`, `WIFI_PASSWORD`), server URL (`SERVER_URL`), and API key (`API_KEY`)

### ML Model Evaluation

Model evaluation scripts are available in the `ml/` directory:

```bash
# Evaluate crop prediction model
python ml/evaluate.py ml/models/random_forest_model.pkl data/cpdata.csv

# Preprocess data
python ml/preprocess.py data/cpdata.csv data/processed_data.csv
```

## Deployment

For production deployment, use a WSGI server:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Set environment variables for production (`FLASK_ENV=production`, `DEBUG=False`) and configure MongoDB with authentication and replica sets. Set up a reverse proxy (nginx/Apache) for SSL termination.

## Troubleshooting

**MongoDB Connection Issues**: Verify MongoDB is running (`mongosh` or check service status), ensure `DB_URI` in `.env` matches your setup, check network connectivity for remote MongoDB.

**Model Not Found Errors**: Train models with `python ml/train_model.py data/cpdata.csv`, verify `MODEL_PATH` in config points to correct location. System uses rule-based fallback if model unavailable.

**API Key Authentication Errors**: Ensure `X-API-Key` header is included in requests, verify API key matches `API_KEY` in `.env`.

**Import Errors**: Activate virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`), reinstall dependencies with `pip install -r backend/requirements.txt`.

**Port Already in Use**: Change port via `export PORT=5001` or update in `.env`, kill existing process if needed.

## Optional Enhancements

If needed, you could add:

- Starter test coverage for core API endpoints and services
- Dev-only Docker Compose setup for local MongoDB and Flask app
- Published OpenAPI/Swagger spec for interactive API documentation
- Sample seed script to populate initial crop data and test sensor readings
- Alerting hooks for out-of-range environmental conditions (email/webhook)
- CI/CD pipeline for automated testing and deployment
- Web dashboard for real-time sensor data visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
