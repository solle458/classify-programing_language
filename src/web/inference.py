"""Web推論機能"""
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from ..models.base import BaseModel
from ..data.preprocessor import PreprocessorFactory


class WebInference:
    """Web用推論クラス"""
    
    def __init__(self, model: BaseModel, preprocessor: Any):
        self.model = model
        self.preprocessor = preprocessor
        
    def predict_single_text(self, text: str) -> Dict[str, Any]:
        """単一テキストの推論"""
        start_time = time.time()
        
        try:
            # 前処理
            processed_text = self.preprocessor.transform([text])
            
            # 推論実行
            predictions = self.model.predict(processed_text)
            probabilities = None
            
            # 確率取得（可能な場合）
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(processed_text)[0]
                except:
                    probabilities = None
            
            end_time = time.time()
            
            # 結果整形
            result = {
                "predicted_language": predictions[0] if len(predictions) > 0 else "Unknown",
                "processing_time": end_time - start_time,
                "success": True
            }
            
            # 確率情報があれば追加
            if probabilities is not None:
                # クラス名取得
                classes = self.model.model.classes_ if hasattr(self.model.model, 'classes_') else None
                if classes is not None:
                    # 上位3つの予測結果
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    top_predictions = [
                        {
                            "language": classes[i],
                            "confidence": float(probabilities[i])
                        }
                        for i in top_indices
                    ]
                    result["top_predictions"] = top_predictions
                    result["all_probabilities"] = {
                        classes[i]: float(probabilities[i]) 
                        for i in range(len(classes))
                    }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


def validate_file_extension(filename: str) -> bool:
    """ファイル拡張子のバリデーション"""
    allowed_extensions = {
        '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', 
        '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r', 
        '.sql', '.sh', '.bat', '.html', '.css', '.xml', '.json', 
        '.yaml', '.yml', '.md', '.txt'
    }
    
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """ファイルサイズのバリデーション"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes
