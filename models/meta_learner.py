"""Meta-öğrenici: sklearn uyumlu PyTorch tabanlı Hızlı Quantile Regressor.

Koenker & Bassett (1978) pinball loss fonksiyonunu gradient descent (Adam) ile optimize eder.
1 Milyon+ satırlık veri kümelerinde doğrusal programlama (LP) çözücülerinin (HiGHS)
$O(N^3)$ karmaşıklık darboğazını aşarak $O(N)$ karmaşıklıkta saniyeler içinde çözüm sunar.
"""

import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class FastQuantileRegressor(BaseEstimator, RegressorMixin):
    """PyTorch tabanlı, Gradient Descent ile çalışan Hızlı Doğrusal Quantile Regressor.
    
    Büyük veri kümelerinde saniyeler içinde doğrusal quantile regresyon eğitimi sağlar.
    Convex optimizasyon teorisi gereği, gradient descent küresel minimuma yakınsar.
    """
    def __init__(self, quantile=0.5, max_iter=1500, lr=0.01, tolerance=1e-6, device='cpu'):
        self.quantile = quantile
        self.max_iter = max_iter
        self.lr = lr
        self.tolerance = tolerance
        self.device = device
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Girdileri numpy array formatına getir ve sayısal tipe (float) zorla
        X_arr = X.astype(float).values if isinstance(X, pd.DataFrame) else np.array(X, dtype=float)
        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)

        # PyTorch Tensor dönüşümü
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        n_features = X_tensor.shape[1]
        
        # Doğrusal katman: y = X*W + b
        model = nn.Linear(n_features, 1).to(self.device)
        
        # Başlangıç ağırlıkları: Eşit ağırlıklı ortalama (Stacking için ideal bir başlangıç noktası)
        nn.init.constant_(model.weight, 1.0 / n_features)
        nn.init.constant_(model.bias, 0.0)

        # Adam optimizer + Cosine Annealing Learning Rate Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=1e-5)

        best_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            preds = model(X_tensor)
            diff = y_tensor - preds
            
            # Pinball Loss (Asimetrik Mutlak Hata) formülasyonu
            loss = torch.mean(torch.max(self.quantile * diff, (self.quantile - 1.0) * diff))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            
            # Tolerans bazlı erken durdurma (Early Stopping)
            if abs(best_loss - loss_val) < self.tolerance:
                no_improve_epochs += 1
                if no_improve_epochs >= 50:
                    break
            else:
                no_improve_epochs = 0
            
            if loss_val < best_loss:
                best_loss = loss_val

        # Katsayıları sklearn formatına çekip kaydet
        self.coef_ = model.weight.detach().cpu().numpy().flatten()
        self.intercept_ = float(model.bias.detach().cpu().numpy()[0])
        return self

    def predict(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return X_arr @ self.coef_ + self.intercept_


def train_meta_learner(
    x_meta: pd.DataFrame,
    y: pd.Series,
    quantiles: list[float] | None = None,
    max_iter: int = 1500,
    lr: float = 0.01,
) -> dict[float, FastQuantileRegressor]:
    """PyTorch tabanlı FastQuantileRegressor meta-öğrenici eğitimi.

    Parameters
    ----------
    x_meta : pd.DataFrame
        Base model OOF tahminleri.
    y : pd.Series
        Normalize edilmiş hedef (y_norm = power_kW / capacity_kW).
    quantiles : list[float]
        Default: [0.1, 0.5, 0.9]

    Returns
    -------
    dict[float, FastQuantileRegressor]
        Her quantile için ayrı eğitilmiş model.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    models: dict[float, FastQuantileRegressor] = {}

    for q in quantiles:
        t0 = time.time()
        logger.info(f"FastQuantileRegressor eğitiliyor: q={q}, max_iter={max_iter}, lr={lr}")
        qr = FastQuantileRegressor(quantile=q, max_iter=max_iter, lr=lr)
        qr.fit(x_meta, y)
        elapsed = time.time() - t0
        models[q] = qr
        logger.info(f"  q={q} modeli eğitildi ({elapsed:.2f}s). Intercept: {qr.intercept_:.4f}")

    return models


def fit_affine_center(y_pred_q05: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Merkez (q05) üzerinden ortak (a,b) katsayılarını en küçük karelerle (Least Squares) çözer."""
    pos = y_true > 0
    if pos.sum() < 10:
        return 1.0, 0.0
    A = np.vstack([y_pred_q05[pos], np.ones(pos.sum())]).T
    a, b = np.linalg.lstsq(A, y_true[pos], rcond=None)[0]
    return float(a), float(b)


def predict_intervals(
    models: dict[float, FastQuantileRegressor],
    x_meta: pd.DataFrame,
    calibration_data: tuple[pd.DataFrame, pd.Series] | None = None,
    calibration_threshold: float = 0.20,
    enforce_monotonicity: bool = True,
) -> pd.DataFrame:
    """Quantile tahminleri üret, opsiyonel olarak koşullu otomatik kalibrasyon uygula.

    Parameters
    ----------
    models : dict[float, FastQuantileRegressor]
        train_meta_learner çıktısı.
    x_meta : pd.DataFrame
        Test set meta-input (12 sütun: 9 base pred + 3 missingness flags).
    calibration_data : tuple[pd.DataFrame, pd.Series], optional
        (x_meta_cal, y_cal) — hedef santralin ilk N gününe ait kalibrasyon verisi.
    calibration_threshold : float
        Sistematik sapma eşiği (|a - 1.0| > threshold ise kalibre edilir).
    enforce_monotonicity : bool
        True ise post-sort ile q01 ≤ q05 ≤ q09 garanti edilir (Karar 8).

    Returns
    -------
    pd.DataFrame
        Sütunlar: q_0.1, q_0.5, q_0.9 (normalize edilmiş tahminler).
    """
    quantiles_sorted = sorted(models.keys())
    col_map = {q: f"q_{q}" for q in quantiles_sorted}

    # 1. Ham (uncalibrated) tahminleri üret
    preds_dict = {}
    for q in quantiles_sorted:
        # PyTorch model predict matris çarpımını el ile çalıştır (segfault güvenliği)
        m = models[q]
        X_arr = x_meta.astype(float).values
        preds_dict[col_map[q]] = X_arr @ m.coef_ + m.intercept_
        
    preds = pd.DataFrame(preds_dict, index=x_meta.index)

    # 2. Koşullu Otomatik Kalibrasyon (Few-Shot Calibration)
    if calibration_data is not None:
        x_cal, y_cal = calibration_data
        
        # Kalibrasyon setinde ham q05 tahmini üret
        m_q05 = models[0.5]
        X_cal_arr = x_cal.astype(float).values
        preds_cal_q05 = X_cal_arr @ m_q05.coef_ + m_q05.intercept_
        
        # q05 üzerinden kalibrasyon katsayılarını (a, b) bul
        a, b = fit_affine_center(preds_cal_q05, y_cal.values)
        
        # Eğer sapma threshold'dan büyükse, band-preserving olarak tüm quantiles dönüştür
        if abs(a - 1.0) > calibration_threshold:
            logger.info(f"Sistematik sapma saptandı (|{a:.4f} - 1.0| > {calibration_threshold:.2f}). Otomatik kalibrasyon uygulanıyor.")
            for q in quantiles_sorted:
                col = col_map[q]
                preds[col] = np.clip(a * preds[col].values + b, 0.0, None)
        else:
            logger.info(f"Sistematik sapma sınırda (|{a:.4f} - 1.0| <= {calibration_threshold:.2f}). Kalibrasyon atlandı.")

    # 3. Monotonluk Garantisi (Karar 8)
    if enforce_monotonicity:
        # Satır bazında artan sıralama garantisi (Chernozhukov 2010)
        vals_sorted = np.sort(preds[[col_map[q] for q in quantiles_sorted]].values, axis=1)
        preds = pd.DataFrame(vals_sorted, columns=[col_map[q] for q in quantiles_sorted], index=preds.index)

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
