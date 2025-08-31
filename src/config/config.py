from dataclasses import dataclass, field
from typing import Dict, Any
import yaml

@dataclass
class DataConfig:
    dataset_name: str = "mnist"
    batch_size: int = 32
    validation_split: float = 0.2
    normalize: bool = True
    min_samples_per_class: int = 10
    lightweight: bool = False  # 軽量化モードフラグ

@dataclass
class ModelConfig:
    model_type: str = "logistic_regression"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 1000,
        "solver": "saga",
        "n_jobs": -1
    })

@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    early_stopping: bool = True
    patience: int = 5

@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "experiments/logs/experiment.log"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment_name: str = "default"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # ネストした設定の処理
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            experiment_name=config_dict.get('experiment_name', 'default'),
            random_seed=config_dict.get('random_seed', 42)
        )
