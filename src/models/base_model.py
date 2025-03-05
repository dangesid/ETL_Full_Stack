import pickle
from abc import ABC, abstractclassmethod

class BaseFraudModel(ABC):
    """ Abstract base class for all fraud detection models. """

    @abstractclassmethod
    def train(self, X_train, y_train):
        """ Train the model."""
        pass

    @abstractclassmethod
    def predict(self, X_test):
        """ Make predictions using the trained model."""
        pass

    def save_model(self, file_path):
        """ Save the trained model in a file"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, file_path):
        """ Load a trained_model from a file"""
        with open(file_path, "rb") as f:
            return pickle.load(f)
        
