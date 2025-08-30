from abc import ABC, abstractmethod
from typing import Any, Dict
import joblib
import numpy as np

class BaseModel(ABC):
    """モデルの着てクラス"""
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        """モデルを保存"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        joblib.dump(self.model, path)

    def load(self, path: str) -> 'BaseModel':
        """モデルを読み込み"""
        self.model = joblib.load(path)
        self.is_fitted = True
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        if not self.is_fitted:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "model_type": self.__class__.__name__,
        }
        
        # sklearn系モデルの場合、追加情報を取得
        if hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_
        if hasattr(self.model, 'classes_'):
            info["n_classes"] = len(self.model.classes_)
            info["classes"] = self.model.classes_.tolist()
            
        return info