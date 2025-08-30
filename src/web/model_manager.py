"""モデル管理機能"""
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..models.classifier import LogisticRegressionModel
from ..data.preprocessor import PreprocessorFactory
from ..data.loader import DataLoaderFactory


class ModelManager:
    """モデルとその前処理器を管理するクラス"""
    
    def __init__(self, model_info_path: str = "models_registry/model_info.json"):
        self.model_info_path = model_info_path
        self.loaded_models = {}
        self.loaded_preprocessors = {}
    
    def load_model_info(self) -> Dict[str, Any]:
        """モデル情報を読み込み"""
        with open(self.model_info_path, "r") as f:
            return json.load(f)
    
    def get_model_and_preprocessor(self, model_id: str) -> tuple:
        """モデルと対応する前処理器を取得"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.loaded_preprocessors[model_id]
        
        model_info = self.load_model_info()
        model_data = next(
            (model for model in model_info["models"] if model["id"] == model_id), 
            None
        )
        
        if not model_data:
            raise ValueError(f"Model {model_id} not found")
        
        # モデルファイルが存在しない場合は再構築
        if not Path(model_data["file_path"]).exists():
            self._ensure_model_exists()
        
        # 新しい形式でモデル読み込み（モデル＋ベクトライザー）
        try:
            model_container = joblib.load(model_data["file_path"])
            if isinstance(model_container, dict):
                # 新形式: モデルとベクトライザーが一緒に保存されている
                sklearn_model = model_container['model']
                vectorizer = model_container['vectorizer']
                
                # LogisticRegressionModelでラップ
                model = LogisticRegressionModel()
                model.model = sklearn_model
                model.is_fitted = True
                
                # ベクトライザーを前処理器として使用
                preprocessor = vectorizer
                
            else:
                # 旧形式: モデルのみ
                model = LogisticRegressionModel()
                model.model = model_container
                model.is_fitted = True
                
                # 前処理器を再構築
                preprocessor = self._rebuild_preprocessor(model_data)
        
        except Exception as e:
            # 読み込み失敗時は再構築
            print(f"モデル読み込み失敗、再構築します: {e}")
            self._ensure_model_exists()
            return self.get_model_and_preprocessor(model_id)
        
        # キャッシュ
        self.loaded_models[model_id] = model
        self.loaded_preprocessors[model_id] = preprocessor
        
        return model, preprocessor
    
    def _ensure_model_exists(self):
        """モデルファイルが存在しない場合は再構築"""
        try:
            from models_registry.download_models import ensure_model_exists
            ensure_model_exists()
        except Exception as e:
            print(f"モデル再構築エラー: {e}")
            raise RuntimeError("モデルの準備に失敗しました")
    
    def _rebuild_preprocessor(self, model_data: Dict[str, Any]) -> Any:
        """訓練データから前処理器を再構築"""
        try:
            # データローダーで訓練データを取得
            data_loader = DataLoaderFactory.create_loader(
                "programming_language", 
                min_samples_per_class=200  # モデル訓練時と同じ設定
            )
            y_train, X_train, _, _ = data_loader.load()
            
            # 前処理器を作成し、訓練データで学習
            preprocessor = PreprocessorFactory.create_preprocessor(
                "programming_language", 
                normalize=False
            )
            preprocessor.fit(X_train)
            
            return preprocessor
            
        except Exception as e:
            raise RuntimeError(f"Failed to rebuild preprocessor: {e}")
    
    def save_preprocessor(self, model_id: str, preprocessor: Any) -> None:
        """前処理器を保存（将来の改善用）"""
        preprocessor_path = f"models_registry/{model_id}_preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
    
    def load_preprocessor(self, model_id: str) -> Optional[Any]:
        """保存済み前処理器を読み込み（将来の改善用）"""
        preprocessor_path = f"models_registry/{model_id}_preprocessor.joblib"
        if Path(preprocessor_path).exists():
            return joblib.load(preprocessor_path)
        return None
