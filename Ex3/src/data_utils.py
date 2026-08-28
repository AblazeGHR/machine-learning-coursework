from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """将不同来源的Titanic字段统一成Kaggle风格。"""
    rename_map = {
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
    }
    df = df.rename(columns=rename_map)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段: {missing}")
    return df


def _load_local_csv() -> pd.DataFrame | None:
    """优先读取本地 Kaggle train.csv。"""
    local_train = DATA_DIR / "train.csv"
    if local_train.exists():
        return pd.read_csv(local_train)

    local_openml_cache = DATA_DIR / "titanic_openml.csv"
    if local_openml_cache.exists():
        return pd.read_csv(local_openml_cache)
    return None


def _load_openml_and_cache() -> pd.DataFrame:
    """若本地没有数据，尝试从 OpenML 拉取并缓存。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(name="titanic", version=1, as_frame=True)
    df = dataset.frame.copy()
    cache_path = DATA_DIR / "titanic_openml.csv"
    df.to_csv(cache_path, index=False)
    return df


def load_titanic_dataset() -> pd.DataFrame:
    """加载并清洗 Titanic 数据集。"""
    df = _load_local_csv()
    if df is None:
        df = _load_openml_and_cache()

    df = _normalize_schema(df)

    # Survived 统一转成 0/1 整型
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """拆分特征和标签。"""
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y

