"""STAGE-6: Meta-Learner Eğitimi.

Karar 2: QuantileRegressor (HiGHS LP, alpha=0.0) seçimi.
"""

import logging
import pathlib
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from models.meta_learner import (
    train_meta_learner,
    predict_intervals,
    compute_coverage,
    compute_pinball,
)


def main() -> None:
    processed_dir = pathlib.Path("data/processed")
    x_meta_path = processed_dir / "x_meta_v2.joblib"
    features_path = processed_dir / "features_v2.parquet"
    out_path     = processed_dir / "meta_models_v2.joblib"

    logger.info("=== STAGE-6: Meta-Learner Eğitimi ===")

    # 1. OOF meta-input yükle
    logger.info(f"OOF matrisi yükleniyor: {x_meta_path}")
    bundle: dict = joblib.load(x_meta_path)
    x_meta_base: pd.DataFrame = bundle["x_meta"]
    y_meta : pd.Series    = bundle["y_meta"]
    val_indices = bundle["val_indices"]

    # Karar 5: Missingness bayraklarını features_v2.parquet'ten oku ve birleştir
    logger.info(f"Eksiklik bayrakları features matrisinden yükleniyor: {features_path}")
    features = pd.read_parquet(features_path, columns=['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing'])
    flags_val = features.iloc[val_indices].reset_index(drop=True)
    
    # x_meta dizin indekslerini sıfırla ve birleştir
    x_meta_base_reset = x_meta_base.reset_index(drop=True)
    x_meta = pd.concat([x_meta_base_reset, flags_val], axis=1)
    x_meta.index = val_indices # İndeksi geri ata

    logger.info(f"  x_meta shape (bayraklar dahil) : {x_meta.shape}")
    logger.info(f"  y_meta shape                   : {y_meta.shape}")
    logger.info(f"  Sütunlar                       : {list(x_meta.columns)}")

    # 2. Meta-model eğitimi (PyTorch FastQuantileRegressor kullanır)
    models = train_meta_learner(
        x_meta=x_meta,
        y=y_meta,
        quantiles=[0.1, 0.5, 0.9],
        max_iter=1500,
        lr=0.01,
    )

    # 3. OOF tahminleri ve metrik hesapla
    logger.info("OOF tahminleri hesaplanıyor...")
    preds_oof = predict_intervals(models, x_meta, enforce_monotonicity=True)

    # Quantile crossing sayısı
    for q in [0.1, 0.5, 0.9]:
        col = f"q_{q}"
        pb  = compute_pinball(y_meta.values, preds_oof[col].values, q)
        logger.info(f"  OOF Pinball loss (q={q}) (Ağırlıksız): {pb:.6f}")

    picp = compute_coverage(y_meta, preds_oof)
    logger.info(f"  OOF PICP [0.1–0.9] (Ağırlıksız) : {picp:.4f}  (beklenen ≈ 0.80)")

    # Quantile crossing kontrolü
    cross_10_50 = (preds_oof["q_0.1"] > preds_oof["q_0.5"]).sum()
    cross_50_90 = (preds_oof["q_0.5"] > preds_oof["q_0.9"]).sum()
    logger.info(f"  Quantile crossing (q10>q50): {cross_10_50}  (enforce_monotonicity=True → 0 beklenir)")
    logger.info(f"  Quantile crossing (q50>q90): {cross_50_90}  (enforce_monotonicity=True → 0 beklenir)")

    # Katsayı tablosu
    logger.info("\n--- Meta-Learner Katsayıları ---")
    for q, mdl in models.items():
        coef_dict = dict(zip(x_meta.columns, mdl.coef_.round(4)))
        logger.info(f"  q={q} | intercept={mdl.intercept_:.4f} | coefs={coef_dict}")

    # 4. Kaydet
    payload = {
        "models"    : models,
        "quantiles" : [0.1, 0.5, 0.9],
        "columns"   : list(x_meta.columns),
        "oof_metrics": {
            "pinball_q10": compute_pinball(y_meta.values, preds_oof["q_0.1"].values, 0.1),
            "pinball_q50": compute_pinball(y_meta.values, preds_oof["q_0.5"].values, 0.5),
            "pinball_q90": compute_pinball(y_meta.values, preds_oof["q_0.9"].values, 0.9),
            "picp_80"    : picp,
        },
    }
    joblib.dump(payload, out_path, compress=3)
    logger.info(f"\nMeta modeller kaydedildi: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
    logger.info("=== STAGE-6 TAMAMLANDI ===")


if __name__ == "__main__":
    main()
