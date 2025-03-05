import numpy as np 
from .base_model import BaseFraudModel
from .anomaly_detection import IsolationForestModel
from .supervised_learning import RandomForestModel, XGBoostModel   

class EnsembleFraudModel(BaseFraudModel):

    def __init__(self):
        self.models = {
            "isolation_forest":IsolationForestModel(),
            "RandomForest":RandomForestModel(),
            "XGBoost":XGBoostModel()
        }
    
    def train(self, X_train, y_train):
        for model in self.models.values():
            model.train(X_train, y_train)

        
    def predict(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        return np.round(np.mean(predictions,axis=0))  # Voting in majority 