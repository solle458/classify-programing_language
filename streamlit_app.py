"""Programming Language Classifier - Streamlit Web App"""
import streamlit as st
import json
import sys
from pathlib import Path
import time

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.classifier import LogisticRegressionModel
from src.web.inference import WebInference, validate_file_extension, validate_file_size
from src.web.model_manager import ModelManager


# ページ設定
st.set_page_config(
    page_title="🤖 Programming Language Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_info():
    """モデル情報を読み込み"""
    with open("models_registry/model_info.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_model_and_preprocessor(model_id: str):
    """モデルと前処理器を読み込み（キャッシュ付き）"""
    try:
        model_manager = ModelManager()
        return model_manager.get_model_and_preprocessor(model_id)
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None, None


def main():
    """メイン処理"""
    
    # ヘッダー
    st.title("🤖 Programming Language Classifier")
    st.markdown("""
    プログラミングコードを入力して、どの言語の可能性が高いかを判定します。
    100種類以上のプログラミング言語に対応しています。
    """)
    
    # モデル情報読み込み
    try:
        model_info = load_model_info()
        active_models = [model for model in model_info["models"] if model["is_active"]]
    except Exception as e:
        st.error(f"モデル情報の読み込みに失敗しました: {e}")
        return
    
    # サイドバー: モデル選択と情報
    with st.sidebar:
        st.header("🤖 モデル選択")
        
        # モデル選択ドロップダウン
        selected_model_id = st.selectbox(
            "使用するモデルを選択",
            options=[model["id"] for model in active_models],
            format_func=lambda x: next(model["name"] for model in active_models if model["id"] == x),
            index=0  # デフォルトは最初のモデル
        )
        
        # 選択されたモデルの詳細情報表示
        selected_model = next(model for model in active_models if model["id"] == selected_model_id)
        
        st.markdown("---")
        st.header("📊 モデル詳細")
        st.write(f"**名前**: {selected_model['name']}")
        st.write(f"**タイプ**: {selected_model['type']}")
        st.write(f"**精度**: {selected_model['accuracy']:.4f}")
        st.write(f"**F1スコア**: {selected_model['f1_score']:.4f}")
        st.write(f"**サイズ**: {selected_model['file_size_mb']:.1f} MB")
        
        with st.expander("📝 説明"):
            st.write(selected_model['description'])
        
        st.markdown("---")
        st.header("⚙️ モデル管理")
        
        # モデル追加機能
        with st.expander("➕ モデル追加"):
            st.write("新しいモデルをアップロードしてください")
            uploaded_model = st.file_uploader(
                "モデルファイル (.joblib)",
                type=['joblib'],
                help="joblib形式で保存されたscikit-learnモデル"
            )
            
            if uploaded_model is not None:
                model_name = st.text_input("モデル名", value=f"Custom Model {len(active_models)+1}")
                model_description = st.text_area("説明", value="ユーザーがアップロードしたカスタムモデル")
                
                if st.button("📤 モデルを追加"):
                    add_custom_model(uploaded_model, model_name, model_description)
        
        # モデル削除機能
        if len(active_models) > 1:  # 最低1つのモデルは残す
            with st.expander("🗑️ モデル削除"):
                model_to_delete = st.selectbox(
                    "削除するモデル",
                    options=[model["id"] for model in active_models if model["id"] != selected_model_id],
                    format_func=lambda x: next(model["name"] for model in active_models if model["id"] == x),
                    help="現在使用中のモデルは削除できません"
                )
                
                if st.button("❌ モデルを削除", type="secondary"):
                    if st.session_state.get('confirm_delete', False):
                        delete_model(model_to_delete)
                        st.rerun()
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ 本当に削除しますか？もう一度ボタンを押してください")
        
        st.markdown("---")
        st.markdown("**📁 対応ファイル形式**")
        st.markdown(".py .js .java .cpp .c .h .cs .php .rb .go .rs .swift .kt .scala .r .sql .html .css .xml .json .yaml .md .txt など")
    
    # 選択されたモデルと前処理器読み込み
    with st.spinner(f"🔄 {selected_model['name']} を読み込み中..."):
        model, preprocessor = load_model_and_preprocessor(selected_model_id)
        if model is None or preprocessor is None:
            st.error("モデルの読み込みに失敗しました")
            return
    
    # 推論エンジン初期化
    inference_engine = WebInference(model, preprocessor)
    
    # メインエリア：推論インターフェース
    st.header("🔍 コード分析")
    
    # タブで入力方式を分ける
    tab1, tab2 = st.tabs(["📁 ファイルアップロード", "✏️ テキスト入力"])
    
    with tab1:
        st.subheader("ファイルからコードを読み込み")
        uploaded_file = st.file_uploader(
            "プログラミングファイルを選択してください",
            type=None,  # 全ファイル許可（後でバリデーション）
            help="対応形式: .py, .js, .java, .cpp など（最大10MB）"
        )
        
        if uploaded_file is not None:
            # ファイルバリデーション
            if not validate_file_extension(uploaded_file.name):
                st.error("❌ 対応していないファイル形式です")
                return
            
            if not validate_file_size(uploaded_file.size):
                st.error("❌ ファイルサイズが10MBを超えています")
                return
            
            # ファイル内容読み取り
            try:
                content = uploaded_file.read().decode('utf-8')
                st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
                
                # ファイル内容プレビュー
                with st.expander("📄 ファイル内容プレビュー"):
                    st.code(content[:1000] + ("..." if len(content) > 1000 else ""), language="text")
                
                # 推論実行
                if st.button("🚀 言語を判定", key="file_predict"):
                    predict_and_display(inference_engine, content, uploaded_file.name)
                    
            except UnicodeDecodeError:
                st.error("❌ ファイルの文字エンコーディングが対応していません（UTF-8のみ対応）")
            except Exception as e:
                st.error(f"❌ ファイル読み込みエラー: {e}")
    
    with tab2:
        st.subheader("テキストを直接入力")
        text_input = st.text_area(
            "プログラミングコードを入力してください",
            height=200,
            placeholder="例:\ndef hello_world():\n    print('Hello, World!')"
        )
        
        if text_input.strip():
            if st.button("🚀 言語を判定", key="text_predict"):
                predict_and_display(inference_engine, text_input, "テキスト入力")


def add_custom_model(uploaded_file, model_name: str, description: str):
    """カスタムモデルを追加"""
    import uuid
    import os
    
    try:
        # 一意のIDを生成
        model_id = f"custom_{uuid.uuid4().hex[:8]}"
        model_filename = f"{model_id}.joblib"
        model_path = f"models_registry/{model_filename}"
        
        # ファイルサイズを取得
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # ファイルを保存
        with open(model_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # モデル情報を更新
        model_info = load_model_info()
        new_model = {
            "id": model_id,
            "name": model_name,
            "type": "custom",
            "file_path": model_path,
            "accuracy": 0.0,  # 未知
            "f1_score": 0.0,  # 未知
            "file_size_mb": round(file_size_mb, 2),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "is_active": True,
            "description": description
        }
        
        model_info["models"].append(new_model)
        
        # 更新されたモデル情報を保存
        with open("models_registry/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        st.success(f"✅ モデル '{model_name}' が正常に追加されました！")
        st.info("ページを更新してモデル一覧を確認してください")
        
    except Exception as e:
        st.error(f"❌ モデル追加エラー: {e}")


def delete_model(model_id: str):
    """モデルを削除"""
    import os
    
    try:
        # モデル情報を読み込み
        model_info = load_model_info()
        
        # 削除対象モデルを特定
        model_to_delete = next((model for model in model_info["models"] if model["id"] == model_id), None)
        if not model_to_delete:
            st.error("削除対象のモデルが見つかりません")
            return
        
        # ファイルを削除
        if os.path.exists(model_to_delete["file_path"]):
            os.remove(model_to_delete["file_path"])
        
        # モデル情報から削除
        model_info["models"] = [model for model in model_info["models"] if model["id"] != model_id]
        
        # デフォルトモデルが削除された場合、新しいデフォルトを設定
        if model_info.get("default_model_id") == model_id:
            if model_info["models"]:
                model_info["default_model_id"] = model_info["models"][0]["id"]
        
        # 更新されたモデル情報を保存
        with open("models_registry/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # セッション状態をリセット
        if 'confirm_delete' in st.session_state:
            del st.session_state.confirm_delete
        
        st.success(f"✅ モデル '{model_to_delete['name']}' が削除されました")
        
    except Exception as e:
        st.error(f"❌ モデル削除エラー: {e}")


def predict_and_display(inference_engine: WebInference, code: str, source_name: str):
    """推論実行と結果表示"""
    
    with st.spinner("🤖 分析中..."):
        result = inference_engine.predict_single_text(code)
    
    if not result["success"]:
        st.error(f"❌ 推論エラー: {result['error']}")
        return
    
    # 結果表示エリア
    st.header("📊 分析結果")
    
    # 基本情報
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 予測言語", result["predicted_language"])
    with col2:
        st.metric("⏱️ 処理時間", f"{result['processing_time']:.3f}秒")
    
    # 上位予測結果（確率付き）
    if "top_predictions" in result:
        st.subheader("🏆 上位予測結果")
        for i, pred in enumerate(result["top_predictions"], 1):
            confidence_percent = pred["confidence"] * 100
            st.write(f"**{i}位**: {pred['language']} ({confidence_percent:.2f}%)")
            st.progress(pred["confidence"])
        
        # 全結果（折りたたみ可能）
        with st.expander("📈 全予測結果を表示"):
            all_probs = result["all_probabilities"]
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            for lang, prob in sorted_probs[:20]:  # 上位20位まで表示
                st.write(f"{lang}: {prob*100:.2f}%")
    
    # 分析対象情報
    st.markdown("---")
    st.caption(f"📝 分析対象: {source_name}")


if __name__ == "__main__":
    main()
