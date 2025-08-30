import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_training_history(history: dict, save_path: str = None) -> None:
    """訓練履歴をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')

    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_samples(
    X: np.ndarray, 
    y: np.ndarray,
    predictions: np.ndarray = None,
    n_samples: int = 10,
    image_shape: Tuple[int, int] = (28, 28)) -> None:
    """サンプル画像を表示"""
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))

    for i in range(n_samples):
        if len(X.shape) == 2:
            image = X[i].reshape(image_shape)
        else:
            image = X[i]

        axes[i].imshow(image, cmap='gray')

        title = f'True: {y[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'

        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()