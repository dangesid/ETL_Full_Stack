import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.base_model import BaseFraudModel

#Load the test data 
df_test = pd.read_csv("data/preprocessed_data.csv")
FEATURE_COLUMN = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                  'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'] 
LABEL_COLUMN = "isFraud"    

X_test = df_test[FEATURE_COLUMN]
y_test = df_test[LABEL_COLUMN]

X_test = X_test.copy()
X_test.loc[:,"type"] = X_test["type"].astype("category").cat.codes
X_test = X_test.astype(np.float32)

isolation_forest = BaseFraudModel.load_model("models/isolation_forest.pkl")
rf_model = BaseFraudModel.load_model("models/random_forest.pkl")
xgb_model = BaseFraudModel.load_model("models/xgboost.pkl")
ensemble_model = BaseFraudModel.load_model("models/ensemble_model.pkl")

# Make predictions using the loaded models
predictions = {
    "Isolation Forest": isolation_forest.predict(X_test),
    "Random Forest": rf_model.predict(X_test),
    "XGBoost": xgb_model.predict(X_test),
    "Ensemble": ensemble_model.predict(X_test)
}

for model_name, y_pred in predictions.items():
    print(f"\n Evaluation {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0)}")  
    print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")  
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")  
    print("-" * 50)

