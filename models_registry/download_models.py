"""
モデル自動ダウンロード・再構築スクリプト
Streamlit Cloud デプロイ用
"""
import os
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def rebuild_model():
    """
    訓練データから軽量モデルを再構築
    GitHub 100MB制限対応
    """
    try:
        print("🔄 モデルを再構築中...")
        
        # 必要なモジュールをインポート
        import sys
        sys.path.append('.')
        from src.data.loader import DataLoaderFactory
        from src.models.classifier import LogisticRegressionModel
        
        # 訓練データ読み込み
        print("📥 訓練データを読み込み中...")
        data_loader = DataLoaderFactory.create_loader(
            "programming_language", 
            min_samples_per_class=200
        )
        y_train, X_train, _, _ = data_loader.load()
        
        # 軽量設定でTF-IDFベクトライザーを作成
        print("🔧 TF-IDFベクトライザーを構築中...")
        vectorizer = TfidfVectorizer(
            max_features=5000,  # 特徴量を5000に制限
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
            stop_words='english'
        )
        
        # 前処理
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # 軽量モデル訓練
        print("🤖 軽量モデルを訓練中...")
        model = LogisticRegression(
            max_iter=500,  # イテレーション数を削減
            solver='saga',
            n_jobs=1,  # Cloud環境用
            random_state=42
        )
        model.fit(X_train_tfidf, y_train)
        
        # モデル保存
        model_path = "models_registry/lr_baseline_001.joblib"
        print(f"💾 モデルを保存中: {model_path}")
        
        # モデルとベクトライザーを一緒に保存
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'classes': model.classes_
        }
        
        joblib.dump(model_data, model_path)
        
        # ファイルサイズ確認
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ モデル再構築完了! サイズ: {file_size:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ モデル再構築エラー: {e}")
        return False

def ensure_model_exists():
    """モデルファイルが存在しない場合は再構築"""
    model_path = "models_registry/lr_baseline_001.joblib"
    
    if not os.path.exists(model_path):
        print("📂 モデルファイルが見つかりません。再構築します...")
        return rebuild_model()
    else:
        print("✅ モデルファイルが存在します")
        return True

if __name__ == "__main__":
    ensure_model_exists()
