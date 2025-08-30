import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .config.config import Config
from .data.loader import DataLoaderFactory
from .data.preprocessor import PreprocessorFactory
from .models.classifier import ModelFactory
from .training.trainer import Trainer
from .evaluation.evaluator import Evaluator
from .utils.logger import setup_logging, get_logger


def main():
    # 引数解析
    parser = argparse.ArgumentParser(description="MNIST Classification")
    parser.add_argument('--config', default='configs/default.yaml', 
                       help='Path to config file')
    args = parser.parse_args()
    
    # 設定読み込み
    config = Config.from_yaml(args.config)
    
    # ロギング設定
    setup_logging(config.logging.level, config.logging.log_file)
    logger = get_logger(__name__)
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Using config: {args.config}")
    
    # 実験ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/{config.experiment_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定保存
    config_dict = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "experiment_name": config.experiment_name,
        "random_seed": config.random_seed
    }
    
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    start_time = time.time()
    
    try:
        # データ読み込み
        logger.info("Loading data...")
        data_loader = DataLoaderFactory.create_loader(
            config.data.dataset_name, 
            min_samples_per_class=config.data.min_samples_per_class
        )
        y_train, X_train, y_test, X_test = data_loader.load()
        logger.info(f"Data loaded: train={len(y_train)}, test={len(X_train)}")
        
        # 前処理
        logger.info("Preprocessing data...")
        preprocessor = PreprocessorFactory.create_preprocessor(
            config.data.dataset_name, 
            normalize=config.data.normalize
        )

        # トレーナーの設定
        trainer = Trainer(
            preprocessor=preprocessor,
            validation_split=config.data.validation_split,
            random_seed=config.random_seed
        )
        
        # モデル作成
        logger.info(f"Creating model: {config.model.model_type}")
        model = ModelFactory.create_model(
            config.model.model_type, 
            **config.model.parameters
        )
        
        # 訓練
        logger.info("Training model...")
        trained_model = trainer.train(model, X_train, y_train)
        
        # テストデータの前処理
        x_test_processed = trainer.prepare_test_data(X_test)
        
        # 評価
        logger.info("Evaluating model...")
        evaluator = Evaluator()
        results = evaluator.evaluate_classification(
            trained_model, x_test_processed, y_test
        )
        
        # 結果表示
        metrics = results['metrics']
        logger.info(f"Results - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        # モデル保存
        model_path = experiment_dir / "model.joblib"
        trained_model.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        # 実験結果保存
        end_time = time.time()
        experiment_results = {
            "experiment_name": config.experiment_name,
            "timestamp": timestamp,
            "duration": end_time - start_time,
            "config": config_dict,
            "metrics": metrics,
            "detailed_results": {
                "classification_report": results['classification_report'],
                "confusion_matrix": results['confusion_matrix'].tolist()
            }
        }
        
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"Experiment completed successfully in {end_time - start_time:.2f}s")
        logger.info(f"Results saved to: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
