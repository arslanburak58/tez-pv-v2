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
    out_path     = processed_dir / "meta_models_v2.joblib"

    logger.info("=== STAGE-6: Meta-Learner Eğitimi ===")

    # 1. OOF meta-input yükle
    logger.info(f"OOF matrisi yükleniyor: {x_meta_path}")
    bundle: dict = joblib.load(x_meta_path)
    x_meta: pd.DataFrame = bundle["x_meta"]
    y_oof : pd.Series    = bundle["y_oof"]

    logger.info(f"  x_meta shape : {x_meta.shape}")
    logger.info(f"  y_oof shape  : {y_oof.shape}")
    logger.info(f"  Sütunlar     : {list(x_meta.columns)}")

    # 2. Meta-model eğitimi
    models = train_meta_learner(
        x_meta=x_meta,
        y=y_oof,
        quantiles=[0.1, 0.5, 0.9],
        alpha=0.0,
        solver="highs",
    )

    # 3. OOF tahminleri ve metrik hesapla
    logger.info("OOF tahminleri hesaplanıyor...")
    preds_oof = predict_intervals(models, x_meta, enforce_monotonicity=True)

    # Quantile crossing sayısı (monotonicity enforcement öncesi)
    # Not: post-sort sonrası crossing sıfır olmak zorunda
    for q in [0.1, 0.5, 0.9]:
        col = f"q_{q}"
        pb  = compute_pinball(y_oof.values, preds_oof[col].values, q)
        logger.info(f"  OOF Pinball loss (q={q}): {pb:.6f}")

    picp = compute_coverage(y_oof, preds_oof)
    logger.info(f"  OOF PICP [0.1–0.9]       : {picp:.4f}  (beklenen ≈ 0.80)")

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
            "pinball_q10": compute_pinball(y_oof.values, preds_oof["q_0.1"].values, 0.1),
            "pinball_q50": compute_pinball(y_oof.values, preds_oof["q_0.5"].values, 0.5),
            "pinball_q90": compute_pinball(y_oof.values, preds_oof["q_0.9"].values, 0.9),
            "picp_80"    : picp,
        },
    }
    joblib.dump(payload, out_path, compress=3)
    logger.info(f"\nMeta modeller kaydedildi: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
    logger.info("=== STAGE-6 TAMAMLANDI ===")


if __name__ == "__main__":
    main()
