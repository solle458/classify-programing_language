from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.logger import get_logger

class DataLoader(ABC):
    """データローダーの基底クラス"""

    @abstractmethod
    def load(self) -> Tuple[Any, ...]:
        pass

class ProgrammingLanguageLoader(DataLoader):
    """プログラミング言語データローダー"""
    def __init__(self, min_samples_per_class: int = 10):
        self.min_samples_per_class = min_samples_per_class
        self.logger = get_logger(self.__class__.__name__)
    
    def load(self) -> Tuple[Any, ...]:
        from collections import Counter
        
        rosetta_datasets = load_dataset("christopher/rosetta-code")
        # Convert to regular Python lists to avoid dataset indexing issues
        all_languages = list(rosetta_datasets['train']['language_name'])
        all_code = list(rosetta_datasets['train']['code'])
        
        # Count samples per language
        language_counts = Counter(all_languages)
        
        # Filter out languages with too few samples
        valid_languages = set(lang for lang, count in language_counts.items() 
                            if count >= self.min_samples_per_class)
        # Filter data
        filtered_languages = []
        filtered_code = []
        
        for lang, code in zip(all_languages, all_code):
            if lang in valid_languages:
                filtered_languages.append(lang)
                filtered_code.append(code)
        
        self.logger.info(f"Filtered data: {len(all_languages)} -> {len(filtered_languages)} samples")
        self.logger.info(f"Languages: {len(language_counts)} -> {len(valid_languages)} languages")
        self.logger.info(f"Removed {len(language_counts) - len(valid_languages)} languages with < {self.min_samples_per_class} samples")
        
        # Split into train and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            filtered_code, filtered_languages, 
            test_size=0.1, 
            random_state=42,
            stratify=filtered_languages  # Now we can use stratify since all classes have enough samples
        )
        return y_train_val, X_train_val, y_test, X_test
    
class DataLoaderFactory:
    """データローダーのファクトリクラス"""
    @staticmethod
    def create_loader(dataset_name: str, min_samples_per_class: int = 10) -> DataLoader:
        loaders = {
            "programming_language": ProgrammingLanguageLoader
        }
        if dataset_name not in loaders:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        if dataset_name == "programming_language":
            return loaders[dataset_name](min_samples_per_class=min_samples_per_class)
        else:
            return loaders[dataset_name]()
        