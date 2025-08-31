# 🤖 Programming Language Classifier

プログラミングコードの内容から言語を自動判定する機械学習プロジェクト＆Webアプリケーション

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org)

## 🎯 プロジェクト概要

このプロジェクトは、プログラミングコードのテキストを解析して、どのプログラミング言語で書かれているかを自動判定するシステムです。機械学習モデルの訓練からWebアプリケーションでのデプロイまで、エンドツーエンドのソリューションを提供します。

### ✨ 主な機能

- 📊 **高精度分類**: 100種類以上のプログラミング言語を78-80%の精度で判定
- 🌐 **Webアプリ**: Streamlitベースの直感的なユーザーインターフェース
- 📁 **ファイル対応**: 25種類以上のプログラミングファイル形式をサポート
- ⚙️ **モデル管理**: 複数モデルの選択・追加・削除機能
- ☁️ **クラウド対応**: Streamlit Cloud用に最適化済み
- 🔒 **セキュリティ**: ファイル形式・サイズのバリデーション

### 🎮 デモ

```python
# 入力例
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 出力例
🏆 予測結果: Python (信頼度: 92.5%)
```

## 📊 性能指標

| モデル | 精度 | F1スコア | サイズ | 用途 |
|--------|------|----------|--------|------|
| **軽量版** | 78.71% | 78.58% | 4.2MB | デプロイ・高速推論 |
| **標準版** | 80.84% | 80.59% | 109MB | 高精度・ローカル用 |

**対応言語**: Python, JavaScript, Java, C++, C, Go, Rust, Swift, Kotlin, PHP, Ruby など104言語

## 🚀 クイックスタート

### 前提条件

- Python 3.11以上
- UV（推奨）またはpip

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd classify-programing_language

# 依存関係をインストール
uv sync
# または: pip install -r requirements.txt
```

### Webアプリを起動

```bash
# Streamlitアプリを起動
uv run streamlit run streamlit_app.py
# または: streamlit run streamlit_app.py

# ブラウザで http://localhost:8501 にアクセス
```

## 📁 プロジェクト構成

```
classify-programing_language/
├── streamlit_app.py          # Webアプリのメインファイル
├── src/                      # ソースコード
│   ├── config/              # 設定管理
│   ├── data/                # データ処理・前処理
│   ├── models/              # モデル定義
│   ├── training/            # 訓練ロジック
│   ├── evaluation/          # 評価・メトリクス
│   ├── utils/               # ユーティリティ
│   └── web/                 # Webアプリ固有の機能
├── configs/                 # 設定ファイル
│   ├── default.yaml         # 標準設定
│   └── lightweight.yaml     # 軽量化設定
├── models_registry/         # モデル管理
│   ├── model_info.json      # モデルメタデータ
│   └── *.joblib            # 訓練済みモデル
├── experiments/             # 実験結果
└── requirements.txt         # 依存関係
```

## 🛠️ 使用方法

### 1. Webアプリでの使用

1. **アプリ起動**: `streamlit run streamlit_app.py`
2. **モデル選択**: サイドバーから使用するモデルを選択
3. **入力**: ファイルアップロードまたはテキスト直接入力
4. **判定実行**: 「🚀 言語を判定」ボタンをクリック
5. **結果確認**: 予測結果と信頼度を確認

### 2. 新しいモデルの訓練

```bash
# 標準モデル（高精度）
uv run python -m src.main --config configs/default.yaml

# 軽量モデル（デプロイ用）
uv run python -m src.main --config configs/lightweight.yaml
```

### 3. カスタム設定

`configs/custom.yaml` を作成して独自の設定で訓練:

```yaml
data:
  dataset_name: "programming_language"
  min_samples_per_class: 100
  lightweight: true  # 軽量化モード

model:
  model_type: "logistic_regression"
  parameters:
    max_iter: 1000
    solver: "saga"

experiment_name: "my_custom_model"
```

## 🔧 技術詳細

### アーキテクチャ

- **データセット**: Rosetta Code Dataset（100+言語、80,000サンプル）
- **前処理**: TF-IDF Vectorization（軽量版: 5,000特徴量）
- **モデル**: Logistic Regression（scikit-learn）
- **Webフレームワーク**: Streamlit
- **デプロイ**: Streamlit Cloud

### 軽量化手法

1. **特徴量削減**: 50,000+ → 5,000特徴量（90%削減）
2. **パラメータ最適化**:
   - `max_features=5000`
   - `ngram_range=(1,2)`
   - `max_df=0.95`, `min_df=2`
   - `stop_words='english'`
3. **モデル圧縮**: 109MB → 4.2MB（96%削減）

### ファイル対応形式

**✅ 対応**: `.py .js .java .cpp .c .h .cs .php .rb .go .rs .swift .kt .scala .r .sql .html .css .xml .json .yaml .md .txt` など

**🚫 非対応**: 実行ファイル（`.exe`）、アーカイブ（`.zip`）、システムファイル

## 📈 実験結果

### 最新実験（軽量版）

```
実験名: classify-programing_language_lightweight
日時: 2025-08-30 22:17:34
精度: 78.71%
F1スコア: 78.58%
実行時間: 19.35秒
モデルサイズ: 4.2MB
```

### パフォーマンス比較

| 設定 | 特徴量数 | 精度 | サイズ | 訓練時間 |
|------|----------|------|--------|----------|
| フル | ~50,000 | 80.84% | 109MB | ~20秒 |
| 軽量 | 5,000 | 78.71% | 4.2MB | ~19秒 |

## 🌐 デプロイ

### Streamlit Cloud

1. **GitHub リポジトリ**: コードをGitHubにプッシュ
2. **Streamlit Cloud**: https://share.streamlit.io/ でデプロイ
3. **設定**: 
   - Main file: `streamlit_app.py`
   - Python version: 3.11

### 自動軽量化

初回デプロイ時に、アプリが自動的に軽量モデルを構築します：

```
📂 モデルファイルが見つかりません。再構築します...
🔄 モデルを再構築中...
🤖 軽量モデルを訓練中...
✅ モデル再構築完了! サイズ: 4.2MB
```

## 🤝 開発者向け

### 開発環境セットアップ

```bash
# 開発用依存関係インストール
uv sync --dev

# プリコミットフック設定
pre-commit install

# テスト実行
pytest tests/
```

### モデル追加

新しいモデルを追加する場合：

1. **新形式で保存**: モデル + TF-IDFベクトライザー
2. **フォーマット**:
   ```python
   {
       'model': sklearn_model,
       'vectorizer': tfidf_vectorizer,
       'classes': model.classes_,
       'model_type': 'LogisticRegressionModel'
   }
   ```
3. **Webアプリ**: 自動的に検出・利用可能

### カスタマイズ

- **新しいモデル**: `src/models/classifier.py` に追加
- **前処理**: `src/data/preprocessor.py` を拡張
- **UI改善**: `streamlit_app.py` を修正

## 📚 参考資料

- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Rosetta Code Dataset](https://huggingface.co/datasets/christopher/rosetta-code)

## 📄 ライセンス

MIT License

## 🙋‍♂️ サポート

問題や質問がある場合：

1. **Issues**: GitHubでIssueを作成
2. **Documentation**: このREADMEとコード内のコメント
3. **Logs**: `experiments/logs/` フォルダのログファイル

---

**開発者**: Summer Sprint 2025 Machine Learning Project
**バージョン**: 1.0.0
**最終更新**: 2025-08-30