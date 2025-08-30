from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.base import BaseModel
from ..utils.logger import get_logger

class Evaluator:
    """モデル評価クラス"""
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def evaluate_classification(
        self, 
        model: BaseModel, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        average: str = 'weighted') -> Dict[str, Any]:
        """分類モデルの評価"""
        self.logger.info("Starting model evaluation...")
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0)
        }

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions':y_pred
        }

        self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return results

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None) -> None:
        """混同行列をプロット"""

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()