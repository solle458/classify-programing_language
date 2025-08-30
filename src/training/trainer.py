import logging
from typing import Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split

from ..models.base import BaseModel
from ..data.preprocessor import Preprocessor
from ..utils.logger import get_logger

class Trainer:
    """モデル訓練クラス"""
    def __init__(self, 
                preprocessor: Optional[Preprocessor] = None,
                validation_split: float = 0.1,
                random_seed: int = 42):
        self.preprocessor = preprocessor
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.logger = get_logger(self.__class__.__name__)

    def train(self, model: BaseModel, x_train: np.ndarray, y_train: np.ndarray) -> BaseModel:
        """モデルを訓練"""
        self.logger.info("Starting model training...")
        
        # データの前処理
        if self.preprocessor:
            self.logger.info("Applying preprocessing...")
            x_train = self.preprocessor.fit_transform(x_train)
        
        # 検証用データの分割
        if self.validation_split > 0:
            x_train_split, x_val, y_train_split, y_val = train_test_split(
                x_train, y_train, 
                test_size=self.validation_split, 
                random_state=self.random_seed,
                stratify=y_train
            )
            self.logger.info(f"Split data: train={x_train_split.shape[0]}, val={x_val.shape[0]}")
        else:
            x_train_split, y_train_split = x_train, y_train
            
        # モデルの訓練
        self.logger.info("Training model...")
        model.fit(x_train_split, y_train_split)
        self.logger.info("Model training completed successfully")

        return model
    
    def prepare_test_data(self, x_test: np.ndarray) -> np.ndarray:
        """テストデータの前処理"""
        if self.preprocessor:
            return self.preprocessor.transform(x_test)
        return x_test