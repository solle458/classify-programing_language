from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessor(ABC):
    """前処理の基底クラス"""
    @abstractmethod
    def fit(self, x_train: np.ndarray) -> 'Preprocessor':
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        return self.fit(x_train).transform(x_train)

class ImagePreprocessor(Preprocessor):
    """画像データの前処理"""
    def __init__(self, normalize: bool = True, flatten: bool = True):
        self.normalize = normalize
        self.flatten = flatten

    def fit(self, x_train: np.ndarray) -> 'ImagePreprocessor':
        #画像の場合、fitで学習することは通常ない
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            x = x / 255.0
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        return x

class NaturalLanguagePreprocessor(Preprocessor):
    """自然言語データの前処理"""
    def __init__(self, vectorizer=None, max_length: int = 128):
        if vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                                max_features=5000,    # 特徴量を5000個に制限 ← 主要な軽量化
                                ngram_range=(1, 2),   # 1-gram, 2-gramのみ
                                max_df=0.95,          # 95%以上の文書に出現する語を除外
                                min_df=2,             # 2回未満の語を除外
                                stop_words='english'  # 英語ストップワード除外
                            )
        else:
            self.vectorizer = vectorizer
        self.max_length = max_length

    def fit(self, X_train: List[str]) -> 'NaturalLanguagePreprocessor':
        if self.vectorizer is not None:
            self.vectorizer.fit(X_train)
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        if self.vectorizer is not None:
            return self.vectorizer.transform(X)
        else:
            raise ValueError("vectorizer must be provided")

    def fit_transform(self, X_train: List[str]) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

class StandardPreprocessor(Preprocessor):
    """標準化前処理"""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, x_train: np.ndarray) -> 'StandardPreprocessor':
        self.scaler.fit(x_train)
        return self
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x)

class PreprocessorFactory:
    """前処理のファクトリクラス"""
    @staticmethod
    def create_preprocessor(dataset_name: str, normalize: bool = True) -> Preprocessor:
        if dataset_name == "programming_language":
            return NaturalLanguagePreprocessor()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
