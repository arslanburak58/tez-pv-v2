"""
Meta-öğrenici: sklearn QuantileRegressor (HiGHS LP).

v2'de Ridge yerine QuantileRegressor kullanılır (Karar 2).

STAGE-6'da tam implementasyon yapılacak.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


def train_meta_learner(
    x_meta: pd.DataFrame,
    y: pd.Series,
    quantiles: list[float] | None = None,
    alpha: float = 0.0,
    solver: str = "highs",
) -> dict[float, QuantileRegressor]:
    """sklearn QuantileRegressor (HiGHS LP) meta-öğrenici.

    Parameters
    ----------
    x_meta : pd.DataFrame
        Base model OOF tahminleri + missingness flags.
    y : pd.Series
        Normalize edilmiş hedef (y_norm = power_kW / capacity_kW).
    quantiles : list[float]
        Default: [0.1, 0.5, 0.9]
    alpha : float
        L1 regularization. 0.0 = saf QR.
    solver : str
        'highs' (default), 'highs-ds', 'highs-ipm'.

    Returns
    -------
    dict[float, QuantileRegressor]
        Her quantile için ayrı eğitilmiş model.
    """
    raise NotImplementedError("STAGE-6'da implement edilecek")


def predict_intervals(
    models: dict[float, QuantileRegressor],
    x_meta: pd.DataFrame,
    enforce_monotonicity: bool = True,
) -> pd.DataFrame:
    """Quantile tahminleri üret, opsiyonel olarak crossing'i düzelt.

    Parameters
    ----------
    models : dict[float, QuantileRegressor]
        train_meta_learner çıktısı.
    x_meta : pd.DataFrame
        Test set meta-input.
    enforce_monotonicity : bool
        True ise post-sort ile q01 ≤ q05 ≤ q09 garanti edilir.

    Returns
    -------
    pd.DataFrame
        Sütunlar: q_0.1, q_0.5, q_0.9 (normalize edilmiş tahminler).
    """
    raise NotImplementedError("STAGE-6'da implement edilecek")
