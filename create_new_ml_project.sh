#!/bin/bash

# 新しい機械学習プロジェクト作成スクリプト
# 使用方法: ./create_new_ml_project.sh project-name "Project Description"

set -e

# 引数チェック
if [ $# -lt 1 ]; then
    echo "使用方法: $0 <project-name> [description]"
    echo "例: $0 classify-cifar \"CIFAR-10 classification project\""
    exit 1
fi

PROJECT_NAME=$1
DESCRIPTION=${2:-"Machine learning project based on ML template"}
CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname "$CURRENT_DIR")
NEW_PROJECT_DIR="$PARENT_DIR/$PROJECT_NAME"

echo "🚀 新しいMLプロジェクトを作成します: $PROJECT_NAME"

# プロジェクトディレクトリが既に存在するかチェック
if [ -d "$NEW_PROJECT_DIR" ]; then
    echo "❌ エラー: プロジェクトディレクトリ '$NEW_PROJECT_DIR' が既に存在します"
    exit 1
fi

# 新しいプロジェクトディレクトリを作成
echo "📁 プロジェクトディレクトリを作成中..."
mkdir -p "$NEW_PROJECT_DIR"

# 基本構造をコピー（experiments と .venv は除外）
echo "📋 テンプレートをコピー中..."
rsync -av \
    --exclude='experiments/' \
    --exclude='.venv/' \
    --exclude='.git/' \
    --exclude='uv.lock' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$CURRENT_DIR/" "$NEW_PROJECT_DIR/"

# 新しいプロジェクト用に調整
cd "$NEW_PROJECT_DIR"

# pyproject.toml を更新
echo "⚙️ プロジェクト設定を更新中..."
sed -i '' "s/name = \"classify-mnist\"/name = \"$PROJECT_NAME\"/" pyproject.toml
sed -i '' "s/description = \".*\"/description = \"$DESCRIPTION\"/" pyproject.toml

# README.md を更新
PROJECT_TITLE=$(echo "$PROJECT_NAME" | tr '-' ' ' | sed 's/\b\w/\u&/g')
sed -i '' "s/# MNIST Classification Project/# $PROJECT_TITLE/" README.md
sed -i '' "s/MNISTデータセットを使用した手書き数字分類プロジェクト/$DESCRIPTION/" README.md
sed -i '' "s/classify-MNIST/$PROJECT_NAME/g" README.md

# configs/default.yaml を更新
sed -i '' "s/experiment_name: \"mnist_baseline\"/experiment_name: \"${PROJECT_NAME}_baseline\"/" configs/default.yaml

# 空のディレクトリを作成
mkdir -p experiments/{logs}
mkdir -p models
mkdir -p data/{raw,processed}

# .gitkeep ファイルを作成
touch experiments/.gitkeep
touch models/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep

# 新しいgitリポジトリを初期化
echo "🔧 Gitリポジトリを初期化中..."
git init
git add .
git commit -m "Initial commit: $PROJECT_NAME project created from template"

# uv.lockを削除（プロジェクト固有のため）
rm -f uv.lock

echo ""
echo "✅ プロジェクト作成完了!"
echo "📍 プロジェクトパス: $NEW_PROJECT_DIR"
echo ""
echo "🚀 次のステップ:"
echo "1. cd $NEW_PROJECT_DIR"
echo "2. uv sync  # 依存関係をインストール"
echo "3. データセット用のローダーを src/data/loader.py で実装"
echo "4. モデルを src/models/classifier.py で実装"
echo "5. 設定を configs/default.yaml で調整"
echo "6. uv run python -m src.main --config configs/default.yaml  # 実行"
echo ""
echo "📚 詳細は README.md を参照してください"
