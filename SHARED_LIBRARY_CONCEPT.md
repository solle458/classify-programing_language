# 共通ライブラリ構想

## 概要

複数の機械学習プロジェクトで共通して使用する部品を独立したライブラリとして管理する構想です。

## 推奨プロジェクト構造

```
ml-projects/
├── 00_ml_commons/                  # 共通ライブラリプロジェクト
│   ├── ml_commons/
│   │   ├── __init__.py
│   │   ├── base/                   # 基底クラス群
│   │   │   ├── __init__.py
│   │   │   ├── model.py           # BaseModel
│   │   │   ├── data_loader.py     # DataLoader
│   │   │   └── preprocessor.py    # Preprocessor
│   │   ├── evaluation/             # 共通評価関数
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py         # メトリクス計算
│   │   │   └── visualization.py   # 評価結果可視化
│   │   ├── utils/                  # 共通ユーティリティ
│   │   │   ├── __init__.py
│   │   │   ├── logger.py          # ロギング設定
│   │   │   ├── config.py          # 設定管理基底
│   │   │   └── experiment.py      # 実験管理
│   │   └── datasets/               # 共通データセット処理
│   │       ├── __init__.py
│   │       ├── image.py           # 画像データ共通処理
│   │       └── text.py            # テキストデータ共通処理
│   ├── pyproject.toml
│   └── README.md
│
├── 01_template/                    # プロジェクトテンプレート
│   ├── src/
│   ├── configs/
│   ├── create_new_ml_project.sh
│   └── pyproject.toml
│
├── 10_classify_mnist/              # 個別プロジェクト
├── 20_classify_cifar/
└── 30_nlp_sentiment/
```

## 共通ライブラリの利点

### ✅ メリット
1. **コードの重複排除**: 共通機能の一元管理
2. **品質向上**: 一箇所での改善が全プロジェクトに適用
3. **開発効率**: 新プロジェクトでの開発時間短縮
4. **一貫性**: プロジェクト間でのインターフェース統一

### ⚠️ 注意点
1. **依存関係管理**: バージョン管理の複雑化
2. **破壊的変更**: 共通ライブラリの変更が他プロジェクトに影響
3. **学習コスト**: ライブラリAPIの習得が必要

## 実装段階

### フェーズ1: テンプレート化
- [x] 現在のプロジェクトをテンプレート化
- [x] 新プロジェクト作成スクリプトの開発

### フェーズ2: 共通ライブラリ抽出（将来の発展）
```python
# 共通ライブラリの使用例
from ml_commons.base import BaseModel, DataLoader
from ml_commons.evaluation import Evaluator
from ml_commons.utils import setup_logging, ExperimentManager

# プロジェクト固有の実装
class MNISTLoader(DataLoader):
    def load(self):
        # MNIST固有の処理
        pass

class CNNModel(BaseModel):
    def __init__(self):
        # CNN固有の実装
        pass
```

### フェーズ3: パッケージ管理（さらなる発展）
```toml
# pyproject.toml での共通ライブラリ依存
[dependencies]
ml-commons = {path = "../00_ml_commons", develop = true}
```

## 使い分けガイドライン

### 🎯 テンプレート方式（推奨開始）
- **適用場面**: 学習・実験段階、プロジェクト数が少ない
- **メリット**: シンプル、独立性、学習しやすい
- **実装**: 現在の `create_new_ml_project.sh` を使用

### 📚 共通ライブラリ方式（発展形）
- **適用場面**: プロジェクト数が増加、チーム開発
- **メリット**: スケーラビリティ、保守性
- **実装**: 後日、必要に応じて段階的に導入

## 推奨移行パス

1. **現在**: テンプレート方式で複数プロジェクトを作成
2. **3-5プロジェクト後**: 共通パターンを特定
3. **共通ライブラリ化**: 頻繁に使用される部分を抽出
4. **段階的移行**: 既存プロジェクトを少しずつ移行

## まとめ

**現時点での推奨**: 
- ✅ **テンプレート方式** (`create_new_ml_project.sh`) から開始
- ✅ プロジェクトごとに独立した開発・実験環境
- ✅ 必要に応じて将来的に共通ライブラリ化を検討
