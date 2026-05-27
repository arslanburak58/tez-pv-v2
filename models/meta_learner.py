"""Meta-öğrenici: sklearn QuantileRegressor (HiGHS LP).

v2'de Ridge yerine QuantileRegressor kullanılır (Karar 2).
Koenker & Bassett (1978) pinball loss LP formülasyonu.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

logger = logging.getLogger(__name__)


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
        L1 regularization. 0.0 = saf QR (LP tam çözüm).
    solver : str
        'highs' (default), 'highs-ds', 'highs-ipm'.

    Returns
    -------
    dict[float, QuantileRegressor]
        Her quantile için ayrı eğitilmiş model.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    models: dict[float, QuantileRegressor] = {}

    for q in quantiles:
        logger.info(f"QuantileRegressor eğitiliyor: q={q}, alpha={alpha}, solver={solver}")
        qr = QuantileRegressor(quantile=q, alpha=alpha, solver=solver)
        qr.fit(x_meta, y)
        models[q] = qr
        logger.info(f"  q={q} modeli eğitildi. Katsayılar: {qr.coef_.round(4)}, Intercept: {qr.intercept_:.4f}")

    return models


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
        True ise post-sort ile q01 ≤ q05 ≤ q09 garanti edilir (Karar 8).

    Returns
    -------
    pd.DataFrame
        Sütunlar: q_0.1, q_0.5, q_0.9 (normalize edilmiş tahminler).
    """
    quantiles_sorted = sorted(models.keys())
    col_map = {q: f"q_{q}" for q in quantiles_sorted}

    preds = pd.DataFrame(
        {col_map[q]: models[q].predict(x_meta) for q in quantiles_sorted},
        index=x_meta.index,
    )

    if enforce_monotonicity:
        # Karar 8: satır bazında artan sıralama garantisi
        vals = np.sort(preds.values, axis=1)
        preds = pd.DataFrame(vals, columns=preds.columns, index=preds.index)

    return preds


def compute_coverage(y_true: pd.Series, preds: pd.DataFrame, lo_col: str = "q_0.1", hi_col: str = "q_0.9") -> float:
    """Prediction interval coverage (PICP) hesaplar."""
    lo = preds[lo_col].values
    hi = preds[hi_col].values
    y = y_true.values
    covered = ((y >= lo) & (y <= hi)).mean()
    return float(covered)


def compute_pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball loss hesaplar."""
    r = y_true - y_pred
    return float(np.mean(np.where(r >= 0, q * r, (q - 1.0) * r)))
