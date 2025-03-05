from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
import xgboost as xgb
from .base_model import BaseFraudModel

# Random Forest Model 

class RandomForestModel(BaseFraudModel):
    def __init__(self, n_estimators=100):
        self.model = SklearnRandomForest(n_estimators=n_estimators, random_state=42)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
#XGBoost Model
    
class XGBoostModel(BaseFraudModel):
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    