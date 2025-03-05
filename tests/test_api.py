import json 
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_api_status(client):
    ''' Test if the api is running or not ''' 
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"message":  "Fraud Detection API is running "}

def test_api_predictions(client):
    ''' Test if the fraud detection predictions are right '''
    test_data = {"features": [5000, 1, 0, 100000, 95000, 0, 0, 0]}
    response = client.post("/predict", data=json.dumps(test_data), content_type="application/json")
    assert response.status_code == 200
    assert "fraud_predictions" in response.get_json()