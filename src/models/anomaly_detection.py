from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.decomposition import PCA
from .base_model import BaseFraudModel

class IsolationForestModel(BaseFraudModel):
    def __init__(self, contamination=0.01):
        self.model = SklearnIsolationForest(contamination=contamination, random_state=42)

    def train(self, X_train, y_train=None):   #using unsupervised learning so y_train is None 
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
class PCAModel(BaseFraudModel):
    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)

    def train(self,X_train,y_train=None):
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.transform(X_test)