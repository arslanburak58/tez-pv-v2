"""
Base learner eğitimi (XGBoost, LightGBM, CatBoost) × 3 quantile.

STAGE-5'te tam implementasyon yapılacak.
"""
from typing import Any
import numpy as np
import pandas as pd


def col_name(algo: str, q: float) -> str:
    """Standart sütun ismi: lgbm_q01, catboost_q05, xgboost_q09 vb."""
    return f"{algo}_q{int(round(q*10)):02d}"


def train_base_learner(
    algo: str,
    q: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> Any:
    """Tek bir base learner eğit (algoritma + quantile kombinasyonu)."""
    raise NotImplementedError("STAGE-5'te implement edilecek")


def make_oof_predictions(
    algo: str,
    q: float,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Walk-forward OOF predictions üret (meta-input için)."""
    raise NotImplementedError("STAGE-5'te implement edilecek")
