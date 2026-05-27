"""Base learner eğitimi (XGBoost, LightGBM, CatBoost) × 3 quantile.

Bu modül, XGBoost için özelleştirilmiş pinball loss kayıp fonksiyonu,
LightGBM ve CatBoost yerleşik quantile loss entegrasyonu ve
OOF tahminlerin walk-forward şeklinde üretilmesini içerir.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def col_name(algo: str, q: float) -> str:
    """Standart sütun ismi: lgbm_q01, catboost_q05, xgboost_q09 vb."""
    return f"{algo}_q{int(round(q*10)):02d}"


class XGBoostPinballObj:
    """Pickle / Joblib serializasyonu ile tam uyumlu XGBoost custom objective sınıfı."""
    def __init__(self, q: float):
        self.q = q
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = y_true - y_pred
        # Birinci türev (gradient)
        grad = np.where(residual < 0, 1.0 - self.q, -self.q)
        # İkinci türev (hessian) - sabit pozitif yaklaşımı
        hess = np.ones_like(grad)
        return grad, hess


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball loss (Quantile Loss) hesaplar."""
    residual = y_true - y_pred
    loss = np.where(residual < 0, (q - 1.0) * residual, q * residual)
    return float(np.mean(loss))


def train_base_learner(
    algo: str,
    q: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> Any:
    """Tek bir base learner eğit (algoritma + quantile kombinasyonu)."""
    if params is None:
        params = {}
        
    # Varsayılan makul parametreler (STAGE-7'de Optuna ile optimize edilecek)
    default_params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
    }
    
    # Kullanıcı parametreleriyle varsayılanları birleştir
    cfg = {**default_params, **params}
    
    if algo == "lgbm":
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
            verbose=-1,
        )
        model.fit(X_train, y_train)
        
    elif algo == "catboost":
        # CatBoost parametrelerinde random_state yerine random_seed kullanılır
        seed = cfg.pop("random_state", 42)
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_seed=seed,
            verbose=0,
        )
        model.fit(X_train, y_train)
        
    elif algo == "xgboost":
        model = XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
            objective=XGBoostPinballObj(q),
        )
        model.fit(X_train, y_train)
        
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo}")
        
    return model


def make_oof_predictions(
    algo: str,
    q: float,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[dict[str, np.ndarray]],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward folds üzerinde Out-of-Fold (OOF) predictions üret.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (val_indices, oof_predictions)
        Validation indeksleri ve bunlara karşılık gelen OOF tahminleri.
    """
    oof_preds_list = []
    val_indices_list = []
    
    for fold_idx, fold in enumerate(folds):
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # Modeli eğitle
        model = train_base_learner(algo, q, X_train, y_train, params)
        
        # validation tahminlerini yap
        preds = model.predict(X_val)
        
        oof_preds_list.extend(preds)
        val_indices_list.extend(val_idx)
        
        # Her fold için pinball loss raporla
        loss = pinball_loss(y.iloc[val_idx].values, preds, q)
        logger.info(f"  Fold {fold_idx} | Model: {algo}_q{q} | Pinball Loss: {loss:.5f}")
        
    return np.array(val_indices_list), np.array(oof_preds_list)
