from flask import Flask, request, jsonify
import numpy as np
import pandas as pd 
import pickle
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Initialise Flask App 
app = Flask(__name__)

# Load the model 
# print(f"current working directory is {os.getcwd()}")

model_path = os.path.abspath("models/ensemble_model.pkl")
# print("model path is :",model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError((f"Model file is not found at {model_path}"))

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running "})

@app.route("/predict", methods=["POST"])
def predict():
    ''' Predict if transaction is fraud or not'''
    try:
        data = request.get_json()
        FEATURE_COLUMN = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                          'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        features_df = pd.DataFrame([data["features"]], columns=FEATURE_COLUMN)

        predictions = model.predict(features_df)
        return jsonify({"fraud_predictions": int(predictions[0])})
    except Exception as e:
        return jsonify({"error": str(e)}),400
    
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "TRUE").lower() == "true"
    port = int(os.environ.get("FLASK_PORT", 5000))
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    
    app.run(host=host, port=port, debug=debug_mode)