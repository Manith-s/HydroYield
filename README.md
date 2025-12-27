# Crop Yield Prediction

A Flask-based API system for intelligent crop recommendation and yield prediction in hydroponic farming environments using machine learning and IoT sensor data.

## What This Does

- **Real-time Environmental Monitoring**: Collects and processes sensor data (temperature, humidity, pH, rainfall) from IoT devices
- **Intelligent Crop Recommendation**: ML-powered suggestions for crops best suited to current environmental conditions
- **Yield Prediction**: Advanced estimation of potential crop yields with confidence scoring and prediction intervals
- **Growing Condition Optimization**: Actionable recommendations to improve environmental parameters for specific crops
- **Historical Analysis**: Trend analysis and visualization of environmental parameters over time
- **IoT Integration**: Arduino and ESP8266 firmware for sensor data collection and transmission

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
- Support for multiple IoT devices with device management
- Historical data retrieval with time-range filtering

## Tech Stack

**Backend**
- Flask 2.2.3
- Flask-CORS 3.0.10
- PyMongo 4.3.3
- Python 3.8+

**Machine Learning**
- scikit-learn 1.2.2
- pandas 1.5.3
- numpy 1.24.2
- joblib 1.2.0

**Database**
- MongoDB 4.4+

**Hardware**
- Arduino (C++)
- ESP8266 (C++)

**Other**
- python-dotenv 1.0.0
- gunicorn 20.1.0

## Repository Structure

```
crop_yield_prediction/
├── backend/
│   ├── app.py                    # Flask application entry point
│   ├── config.py                 # Configuration management
│   ├── requirements.txt          # Python dependencies
│   ├── controllers/
│   │   ├── sensor_controller.py  # Sensor data endpoints
│   │   └── prediction_controller.py  # ML prediction endpoints
│   ├── services/
│   │   ├── data_service.py       # Data processing & analytics
│   │   └── ml_service.py         # ML model inference
│   ├── models/
│   │   ├── crop_model.py         # Crop data models
│   │   └── sensor_model.py       # Sensor data models
│   └── utils/
│       └── helpers.py            # Validation & utilities
├── ml/
│   ├── train_model.py            # Crop prediction model training
│   ├── crop_yield_predictor.py    # Yield prediction model
│   ├── preprocess.py             # Data preprocessing
│   ├── evaluate.py               # Model evaluation
│   └── models/
│       └── random_forest_model.pkl  # Trained model files
├── hardware/
│   ├── arduino_code.ino          # Arduino sensor code
│   └── esp8266_code.ino          # ESP8266 WiFi module code
├── data/
│   └── cpdata.csv                # Crop dataset
├── LICENSE                        # MIT License
└── README.md                      # This file
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

#### Sensor Data

**POST `/api/sensor-data`**
- Submit sensor readings from IoT devices
- Body: `{"device_id": "device1", "temperature": 25.5, "humidity": 60.2, "ph": 6.8, "rainfall": 100}`

**GET `/api/sensor-data/<device_id>`**
- Get sensor history for a device
- Query params: `?days=7&limit=100`

**GET `/api/devices`**
- List all registered devices

#### Predictions

**POST `/api/predict`**
- Predict suitable crops for current conditions
- Body: `{"temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 150}`

**POST `/api/predict-yield`**
- Estimate crop yield
- Body: `{"crop": "rice", "temperature": 25, "humidity": 70, "ph": 6.5, "rainfall": 150}`

**GET `/api/optimal-conditions/<crop>`**
- Get optimal growing conditions for a crop

**POST `/api/suggest-improvements`**
- Get suggestions to improve conditions
- Body: `{"crop": "rice", "current_conditions": {"temperature": 30, "humidity": 50, "ph": 5.0}}`

**GET `/api/analyze-crop/<crop_name>`**
- Analyze historical data for a crop
- Query params: `?days=30`

**GET `/api/crops`**
- List all supported crops

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

1. **Arduino Setup**
   - Upload `hardware/arduino_code.ino` to your Arduino
   - Connect sensors: DS18B20 (temperature), pH sensor, humidity sensor
   - Configure pin assignments in the code

2. **ESP8266 Setup**
   - Upload `hardware/esp8266_code.ino` to your ESP8266
   - Update WiFi credentials: `WIFI_SSID`, `WIFI_PASSWORD`
   - Update server URL: `SERVER_URL`
   - Update API key: `API_KEY`
   - Connect Arduino serial output to ESP8266 RX/TX pins

## Testing

Model evaluation scripts are available in the `ml/` directory:

```bash
# Evaluate crop prediction model
python ml/evaluate.py ml/models/random_forest_model.pkl data/cpdata.csv

# Preprocess data
python ml/preprocess.py data/cpdata.csv data/processed_data.csv
```

## Deployment

### Production Deployment

1. **Set environment variables** for production:
   ```env
   FLASK_ENV=production
   DEBUG=False
   ```

2. **Use a production WSGI server**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Configure MongoDB** for production (replica set, authentication, etc.)

4. **Set up reverse proxy** (nginx/Apache) for SSL termination

### Hardware Deployment

- Ensure stable WiFi connection for ESP8266
- Implement error handling and retry logic for network failures
- Consider local data buffering on device for offline operation

## Troubleshooting

**MongoDB Connection Issues**
- Verify MongoDB is running: `mongosh` or check service status
- Check `DB_URI` in `.env` matches your MongoDB setup
- Ensure network connectivity if using remote MongoDB

**Model Not Found Errors**
- Train models first: `python ml/train_model.py data/cpdata.csv`
- Verify `MODEL_PATH` in config points to correct location
- System will use rule-based fallback if model unavailable

**API Key Authentication Errors**
- Ensure `X-API-Key` header is included in requests
- Verify API key matches `API_KEY` in `.env`

**Import Errors**
- Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Reinstall dependencies: `pip install -r backend/requirements.txt`

**Port Already in Use**
- Change port: `export PORT=5001` or update in `.env`
- Kill existing process: `lsof -ti:5000 | xargs kill` (Linux/Mac)

## Roadmap

- [ ] Add unit and integration tests
- [ ] Implement Docker containerization
- [ ] Add API documentation (OpenAPI/Swagger)
- [ ] Implement user authentication and multi-user support
- [ ] Add real-time notifications for out-of-range conditions
- [ ] Create web dashboard for visualization
- [ ] Add automated model retraining pipeline
- [ ] Support for additional sensor types

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
