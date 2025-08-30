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
        
        # テストデータで性能評価
        print("📊 モデル性能を評価中...")
        y_train, X_train, y_test, X_test = data_loader.load()
        X_test_tfidf = vectorizer.transform(X_test)
        
        # 予測と評価
        y_pred = model.predict(X_test_tfidf)
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 精度: {accuracy:.4f}")
        print(f"🎯 F1スコア: {f1:.4f}")
        
        # モデル保存
        model_path = "models_registry/lr_baseline_001.joblib"
        print(f"💾 モデルを保存中: {model_path}")
        
        # モデルとベクトライザーを一緒に保存
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'classes': model.classes_,
            'performance': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'n_features': 5000,
                'n_classes': len(model.classes_)
            }
        }
        
        joblib.dump(model_data, model_path)
        
        # ファイルサイズ確認
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ モデル再構築完了! サイズ: {file_size:.1f}MB")
        
        # model_info.jsonを更新
        update_model_info(accuracy, f1, file_size)
        
        return True
        
    except Exception as e:
        print(f"❌ モデル再構築エラー: {e}")
        return False

def update_model_info(accuracy: float, f1_score: float, file_size_mb: float):
    """model_info.jsonを実際の性能で更新"""
    try:
        import json
        import time
        
        model_info = {
            "models": [
                {
                    "id": "lr_baseline_001",
                    "name": "LR Cloud Optimized (Auto-generated)",
                    "type": "logistic_regression",
                    "file_path": "models_registry/lr_baseline_001.joblib",
                    "accuracy": round(accuracy, 4),
                    "f1_score": round(f1_score, 4),
                    "file_size_mb": round(file_size_mb, 1),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "is_active": True,
                    "description": f"Auto-generated lightweight model (max_features=5000). Accuracy: {accuracy:.2%}"
                }
            ],
            "default_model_id": "lr_baseline_001"
        }
        
        with open("models_registry/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📝 model_info.json更新完了 (精度: {accuracy:.2%})")
        
    except Exception as e:
        print(f"⚠️ model_info.json更新エラー: {e}")

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
