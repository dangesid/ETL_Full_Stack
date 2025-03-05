# Fraud Detection Project

## Overview
This project aims to detect fraudulent transactions using machine learning models, including Isolation Forest, Random Forest, XGBoost, and an Ensemble Model. It features data preprocessing, feature engineering, model training, evaluation, and a Flask API for real-time fraud prediction.

## Project Structure
```
fraud_detection_project/
│── data/                     # Raw and preprocessed data
│── models/                   # Trained models (saved as .pkl files)
│── src/
│    ├── data_preprocessing.py  # Data cleaning and preparation
│    ├── feature_engineering.py # Feature extraction and selection
│    ├── model_training/        # Model implementation
│    │    ├── isolation_forest.py  
│    │    ├── xgboost_model.py  
│    │    ├── lstm_model.py  
│    │    ├── ensemble.py  
│    ├── fraud_detection_api.py  # Flask API for fraud detection
│    ├── kafka_producer.py       # Kafka producer for streaming data
│    ├── kafka_consumer.py       # Kafka consumer to process transactions
│── tests/                     # Unit tests
│── deployment/                # Docker & deployment scripts
│    ├── Dockerfile            # Docker containerization
│    ├── docker-compose.yml    # Docker Compose configuration
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Docker (if running in a container)

### Step 1: Clone the Repository
```bash
git clone https://github.com/dangesid/ETL_Full_Stack.git
cd ETL_Full_Stack/fraud_detection_project
```

### Step 2: Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
### Training the Models
```bash
python src/train_models.py
```
This script will train and save the models in the `models/` directory.

### Running the API
```bash
python src/fraud_detection_api.py
```
The API will be accessible at `http://127.0.0.1:5000/predict`.

### Making a Prediction
Send a POST request to the API with transaction features:
```json
{
  "features": [value1, value2, ..., valueN]
}
```
Example using `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [0.1, 0.5, ..., 0.8]}'
```

### Running Tests
```bash
pytest tests/
```

## Docker Deployment
### Step 1: Build the Docker Image
```bash
docker build -t fraud-detection-api .
```

### Step 2: Run the Container
```bash
docker run -p 5000:5000 fraud-detection-api
```

Now, the API will be accessible at `http://localhost:5000/predict`.

## Kafka Streaming (Optional)
If you are using Kafka for real-time fraud detection:
```bash
python src/kafka_producer.py  # Start Kafka producer
python src/kafka_consumer.py  # Start Kafka consumer
```

## Contributing
Feel free to submit pull requests or report issues on GitHub.

## License
MIT License

