import os 
import pickle
import pytest
import numpy as np
import pandas as pd
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


model_path = os.path.abspath("models/ensemble_model.pkl")

@pytest.fixture
def load_model():
    ''' Test to ensure that the model is loaded correctly '''
    if not os.path.exists(model_path):
        pytest.fail("Model file does not exist")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

def test_model_loading(load_model):
    ''' Test to ensure that the model is loaded correctly '''
    assert load_model is not None, "Failed to load the model"


def test_model_output_type(load_model):
    ''' Test to ensure that the model output type is int value'''

    FEATURE_COLUMNS = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    test_data = pd.DataFrame([[5000, 1, 0, 100000, 95000, 0, 0, 0]], columns=FEATURE_COLUMNS)

    predictions = load_model.predict(test_data)
    prediction_value = int(predictions[0])

    assert isinstance(prediction_value, int), f"Model output is not an integer value: {type(predictions[0])}"