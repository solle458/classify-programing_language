import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .base import BaseModel

class LogisticRegressionModel(BaseModel):
    """ロジスティック回帰モデル"""
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

class RandomForestModel(BaseModel):
    """ランダムフォレストモデル"""
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

class SVMModel(BaseModel):
    """サポートベクトルマシンモデル"""
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVC(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

class ModelFactory:
    """モデルのファクトリクラス"""
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        models = {
            "logistic_regression": LogisticRegressionModel,
            "random_forest": RandomForestModel,
            "svm": SVMModel
        }
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        return models[model_type](**kwargs)