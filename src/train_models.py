import pandas as pd 
from sklearn.model_selection import train_test_split
from models.anomaly_detection import IsolationForestModel
from models.supervised_learning import RandomForestModel, XGBoostModel
from models.ensemble_model import EnsembleFraudModel
from sklearn.preprocessing import LabelEncoder

# Load preprocess dataset 
df = pd.read_csv('data/preprocessed_data.csv')
print(df.columns)

FEATURE_COLUMN = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                  'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
print("Selected features:", FEATURE_COLUMN)

LABEL_COLUMN = "isFraud"
print(df.columns)
X = df[FEATURE_COLUMN]
y = df[LABEL_COLUMN]


Label_encoder = LabelEncoder()
X.loc[:, "type"] = Label_encoder.fit_transform(X["type"])
X["type"] = X["type"].astype("category").cat.codes
X.drop(columns=["nameOrig", "nameDest"], inplace=True, errors="ignore")
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.dtypes)

# Train models 

print("Training Isolation Forest Model.... ")
isolation_forest_model = IsolationForestModel()
isolation_forest_model.train(X_train)

print("Training Random Forest Model.... ")
random_forest_model = RandomForestModel()
random_forest_model.train(X_train, y_train)

print("Training XGBoost Model.... ")
xgboost_model = XGBoostModel()
xgboost_model.train(X_train, y_train)

print("Training Ensemble Model.... ")
ensemble_model = EnsembleFraudModel()
ensemble_model.train(X_train, y_train)

# Save Models
isolation_forest_model.save_model("models/isolation_forest.pkl")
random_forest_model.save_model("models/random_forest.pkl")
xgboost_model.save_model("models/xgboost.pkl")
ensemble_model.save_model("models/ensemble_model.pkl")

print("Models saved successfully!")
