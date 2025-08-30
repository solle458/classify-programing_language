# classify programing_language

Machine learning project based on ML templateです。機械学習のベストプラクティスに従って構築されています。

## プロジェクト構成

```
classify-programing_language/
├── src/                    # ソースコード
│   ├── config/            # 設定管理
│   ├── data/              # データ処理
│   ├── models/            # モデル定義
│   ├── training/          # 訓練ロジック
│   ├── evaluation/        # 評価・メトリクス
│   ├── utils/             # ユーティリティ
│   └── main.py           # エントリーポイント
├── configs/               # 設定ファイル
├── experiments/           # 実験結果
├── models/               # 保存されたモデル
├── tests/                # テストコード
├── pyproject.toml        # 依存関係管理
└── README.md            # このファイル
```

## セットアップ

### 1. 依存関係のインストール

```bash
# uvを使用する場合（推奨）
uv sync

# pipを使用する場合
pip install -e .
```

### 2. 基本的な実行

```bash
# デフォルト設定で実行
python src/main.py

# 特定の設定ファイルで実行
python src/main.py --config configs/experiments/my_experiment.yaml
```

## 設定

設定は `configs/default.yaml` で管理されています：

```yaml
data:
  dataset_name: "mnist"
  batch_size: 32
  validation_split: 0.2
  normalize: true

model:
  model_type: "logistic_regression"
  parameters:
    max_iter: 1000
    solver: "saga"
    n_jobs: -1

experiment_name: "mnist_baseline"
random_seed: 42
```

## 実験の実行

### 1. 基本的な実験

```bash
python src/main.py --config configs/default.yaml
```

### 2. 異なるモデルでの実験

新しい設定ファイルを作成して実験を実行：

```bash
# 新しい実験設定を作成
cp configs/default.yaml configs/experiments/random_forest.yaml

# 設定を編集してからら実行
python src/main.py --config configs/experiments/random_forest.yaml
```

## 開発

### テストの実行

```bash
pytest tests/
```

### コード品質チェック

```bash
# フォーマット
black src/

# リンター
flake8 src/

# 型チェック
mypy src/
```

## 実験結果

実験結果は `experiments/` ディレクトリに保存されます：

```
experiments/
├── mnist_baseline_20241220_143022/
│   ├── config.json        # 使用した設定
│   ├── results.json       # 評価結果
│   └── model.joblib       # 訓練済みモデル
└── ...
```

## トラブルシューティング

### よくある問題

1. **インポートエラー**: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` でパスを設定
2. **設定ファイルが見つからない**: 作業ディレクトリがプロジェクトルートであることを確認
3. **メモリ不足**: `batch_size` を小さくするか、データのサブセットで実験

詳細は [機械学習ガイド](../../machine_learning_guides.md#トラブルシューティング) を参照してください。

## ライセンス

MIT License
