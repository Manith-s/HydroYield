# HydroYield: Smart Hydroponic Crop Yield Prediction System

A comprehensive Flask-based backend system for intelligent crop recommendation and yield prediction in hydroponic farming environments.

## 📋 Project Overview

HydroYield is an intelligent system designed to help hydroponic farmers optimize crop selection and maximize yields through data-driven decisions. The system collects environmental data from IoT sensors, processes it using machine learning models, and provides actionable recommendations for optimal crop selection and growing conditions.

## 🌟 Key Features

- **Real-time Environmental Monitoring**: Collects and processes sensor data (temperature, humidity, pH, etc.)
- **Intelligent Crop Recommendation**: ML-powered suggestions for crops best suited to current conditions
- **Yield Prediction**: Advanced estimation of potential crop yields with confidence scoring
- **Growing Condition Optimization**: Actionable recommendations to improve environmental parameters
- **Historical Analysis**: Trend analysis and visualization of environmental parameters over time

## 🏗️ System Architecture

The system follows a modular, service-oriented architecture designed for scalability and maintainability:

```
                       ┌─────────────────┐
                       │    API Layer    │
                       │  (Controllers)  │
                       └────────┬────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
    ┌───────────▼──────┐ ┌──────▼───────┐ ┌─────▼─────────┐
    │  Sensor Service  │ │ ML Service   │ │  Data Service │
    │ (Data Ingestion) │ │(Predictions) │ │  (Analytics)  │
    └───────────┬──────┘ └──────┬───────┘ └─────┬─────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                       ┌────────▼────────┐
                       │   Data Layer    │
                       │   (MongoDB)     │
                       └─────────────────┘
```

### Component Breakdown:

1. **Controllers**: Handle HTTP requests/responses, input validation, and route to appropriate services
   - `sensor_controller.py`: Handles sensor data ingestion endpoints
   - `prediction_controller.py`: Handles prediction and analysis endpoints

2. **Services**: Contain core business logic and data processing
   - `data_service.py`: Processes sensor data, calculates derived metrics, manages historical data
   - `ml_service.py`: Handles ML model inference, predictions, and recommendations

3. **ML Pipeline**: Manages model training, evaluation, and prediction
   - `train_model.py`: Trains the primary crop recommendation model
   - `crop_yield_predictor.py`: Handles yield prediction functionality
   - `preprocess.py`: Data preprocessing utilities
   - `evaluate.py`: Model evaluation and performance metrics

4. **Utilities**: Supporting functionality
   - `helpers.py`: Input validation, data sanitization, notification handling

5. **Configuration**: System settings and parameters
   - `config.py`: Environment-specific configuration settings, optimal growing parameters

## 🔍 Technical Details

### Backend Framework

The system is built on Flask, a lightweight Python web framework, with the following key components:

- **Flask Blueprints**: Modular routing for different API categories
- **Error Handling**: Comprehensive error catching and reporting
- **Middleware**: API key validation, request logging
- **Dependency Injection**: Configuration and database access via Flask's application context

### Database

MongoDB is used for data storage, with the following collections:
- `sensor_data`: Time-series environmental readings from IoT devices
- Additional collections can be added for user management, device registration, etc.

### Machine Learning Pipeline

The ML pipeline uses scikit-learn for model training and inference:

1. **Crop Prediction Model**: Random Forest classifier with hyperparameter tuning
   - Features: temperature, humidity, pH, rainfall
   - Target: optimal crop type
   - Performance: ~95-99% accuracy (varies by dataset)

2. **Yield Prediction Model**: Random Forest regressor with confidence estimation
   - Features: crop type plus environmental parameters
   - Target: estimated yield in kg/hectare
   - Includes confidence scoring based on prediction variance

3. **Fallback Mechanisms**: Rule-based prediction systems when models are unavailable

### API Endpoints

#### Sensor Data
- `POST /api/sensor-data`: Submit sensor readings from IoT devices
- `GET /api/sensor-data/{device_id}`: Get sensor history for a specific device
- `GET /api/devices`: List all connected devices

#### Predictions
- `POST /api/predict`: Predict suitable crops for current conditions
- `POST /api/predict-yield`: Estimate crop yield based on conditions
- `GET /api/optimal-conditions/{crop}`: Get optimal growing conditions for a crop
- `POST /api/suggest-improvements`: Get suggestions to improve growing conditions
- `GET /api/analyze-crop/{crop_name}`: Get analysis of historical data for a crop
- `GET /api/crops`: List all supported crops

### Data Validation & Processing

The system includes rigorous data validation and processing:

- **Input Validation**: Schema validation for all API endpoints
- **Data Normalization**: Standardization of sensor readings
- **Quality Scoring**: Automated assessment of sensor data reliability
- **Derived Metrics**: Calculation of additional metrics like VPD (Vapor Pressure Deficit)

### Security Features

- **API Key Authentication**: Required for all endpoints
- **Input Sanitization**: Prevention of injection attacks
- **Error Handling**: Prevents information leakage in error responses

## 📊 Machine Learning Model Details

### Crop Recommendation Model

The crop recommendation system uses a Random Forest classifier trained on the `cpdata.csv` dataset with:

- **Features**: temperature, humidity, pH, rainfall
- **Target**: 22 different crop types
- **Performance**:
  - Accuracy: 97-99%
  - Precision: 0.98
  - Recall: 0.97
  - F1 Score: 0.97

Feature importance analysis shows that pH and temperature are the most influential factors in crop selection.

### Yield Prediction Model

The yield prediction model uses both ML and rule-based approaches:

- **ML Approach**: Random Forest regressor trained on synthetic yield data based on optimal growing conditions
- **Rule-Based Fallback**: Formula-based estimation using proximity to optimal growth conditions
- **Confidence Estimation**: Uses the variance of individual tree predictions to estimate prediction reliability

The model provides yield estimates in kg/hectare along with lower and upper bound estimates.

## 🛠️ Technical Requirements

- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher
- **Libraries**:
  - Flask 2.2.3
  - pandas 1.5.3
  - numpy 1.24.2
  - scikit-learn 1.2.2
  - pymongo 4.3.3
  - And others listed in requirements.txt

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hydro-yield.git
   cd hydro-yield
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file with:
   ```
   FLASK_ENV=development
   DB_URI=mongodb://localhost:27017/
   DB_NAME=hydroponic_system
   API_KEY=your-api-key-here
   SECRET_KEY=your-secret-key-here
   ```

5. **Start the application**
   ```bash
   python app.py
   ```

## 📡 Connecting IoT Devices

IoT devices can connect to the system by sending HTTP POST requests to the `/api/sensor-data` endpoint with the following JSON payload:

```json
{
  "device_id": "unique_device_identifier",
  "temperature": 25.5,
  "humidity": 60.2,
  "ph": 6.8,
  "timestamp": "2023-05-01T12:00:00Z"
}
```

The timestamp is optional and will be added by the system if not provided.

## 🧠 ML Model Training

To train the machine learning models:

```bash
# Preprocess the data
python ml/preprocess.py data/cpdata.csv data/processed_data.csv

# Train the crop prediction model
python ml/train_model.py data/processed_data.csv

# Train the yield prediction model
python ml/crop_yield_predictor.py data/cpdata.csv ml/models/yield_prediction_model.pkl
```

## 📈 Future Enhancements

Planned future improvements include:

1. **Real-time Notifications**: Alert system for out-of-range environmental conditions
2. **Automated Control**: Integration with actuators for automatic environment adjustment
3. **Advanced Visualization**: Interactive dashboards for data visualization
4. **User Management**: Multi-user support with role-based access control
5. **Mobile Application**: Companion mobile app for monitoring and control

## 📚 Project Structure

```
HydroYield/
├── backend/
│   ├── controllers/          # API route handlers
│   │   ├── __init__.py
│   │   ├── prediction_controller.py
│   │   └── sensor_controller.py
│   ├── models/               # Data models
│   │   ├── __init__.py
│   │   ├── crop_model.py
│   │   └── sensor_model.py
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   ├── data_service.py
│   │   └── ml_service.py
│   └── utils/                # Helper functions
│       ├── __init__.py
│       └── helpers.py
├── data/
│   └── cpdata.csv            # Crop dataset
├── ml/
│   ├── models/               # Trained model files
│   ├── crop_yield_predictor.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train_model.py
├── app.py                    # Main application entry point
├── config.py                 # Configuration settings
└── requirements.txt          # Dependencies
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

