import logging
import pathlib
import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from models.base_learners import train_base_learner, pinball_loss, col_name, make_oof_predictions
from models.meta_learner import train_meta_learner, predict_intervals, compute_coverage

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Silent Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = pathlib.Path(__file__).parent.parent
FEATURES_FILE = ROOT / "data" / "processed" / "features_v2.parquet"
SPLITS_FILE = ROOT / "data" / "processed" / "splits.joblib"
STUDIES_DIR = ROOT / "data" / "processed" / "optuna_studies"
BASE_MODELS_OUT = ROOT / "data" / "processed" / "base_models_v2.joblib"
X_META_OUT = ROOT / "data" / "processed" / "x_meta_v2.joblib"
META_MODELS_OUT = ROOT / "data" / "processed" / "meta_models_v2.joblib"

def make_composite_objective(algo: str, X: pd.DataFrame, y: pd.Series, folds: list, sample_weight: pd.Series | None):
    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        if algo == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            }
        elif algo == "catboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        elif algo == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
            
        fold_scores = []
        
        # Optimizasyonu hızlandırmak için son 1 fold'u kullan (en son ve en zorlayıcı walk-forward fold)
        for fold_idx, fold in enumerate(folds[-1:]):
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None
            
            # Üç quantile modelini eğit
            models = {}
            preds_dict = {}
            for q in [0.1, 0.5, 0.9]:
                model = train_base_learner(algo, q, X_train, y_train, params, sample_weight=w_train)
                preds_dict[q] = model.predict(X_val)
                
            # Monotonluk düzeltmesi (sorting)
            preds_val = np.column_stack([preds_dict[0.1], preds_dict[0.5], preds_dict[0.9]])
            preds_val_sorted = np.sort(preds_val, axis=1)
            
            # Sadece gündüz ve aktif üretim saatlerini filtrele
            dl_mask = X_val['cos_zenith'] > 0.087
            pos_mask = y_val > 0
            filter_mask = dl_mask & pos_mask
            
            if filter_mask.sum() == 0:
                continue
                
            y_val_filtered = y_val[filter_mask].values
            preds_sorted_filtered = preds_val_sorted[filter_mask]
            
            # Pinball Loss ve Gündüz Kapsama Oranı (Unweighted)
            pb_q10 = pinball_loss(y_val_filtered, preds_sorted_filtered[:, 0], 0.1)
            pb_q50 = pinball_loss(y_val_filtered, preds_sorted_filtered[:, 1], 0.5)
            pb_q90 = pinball_loss(y_val_filtered, preds_sorted_filtered[:, 2], 0.9)
            mean_pb = (pb_q10 + pb_q50 + pb_q90) / 3.0
            
            cov = np.mean((y_val_filtered >= preds_sorted_filtered[:, 0]) & 
                          (y_val_filtered <= preds_sorted_filtered[:, 2]))
            
            # Kompozit Skor Formülasyonu (Ders 9: Kapsama cezası)
            coverage_penalty = 0.10 * abs(cov - 0.80)
            score = mean_pb + coverage_penalty
            fold_scores.append(score)
            
        if len(fold_scores) == 0:
            return float('inf')
            
        return float(np.mean(fold_scores))
        
    return objective

def main():
    logger.info("=== STAGE-7: Optuna Hiperparametre Optimizasyonu ===")
    
    # 1. Girdileri Yükle
    if not FEATURES_FILE.exists() or not SPLITS_FILE.exists():
        raise FileNotFoundError("Gerekli girdi dosyaları bulunamadı. Lütfen STAGE-3 ve STAGE-4'ü tamamlayın.")
        
    df = pd.read_parquet(FEATURES_FILE)
    splits = joblib.load(SPLITS_FILE)
    
    exclude_cols = ["timestamp", "station_id", "power_kW", "y_norm", "original_index", "sample_weight"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df["y_norm"].copy()
    sample_weight = df["sample_weight"].copy() if "sample_weight" in df.columns else None
    
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    
    best_params = {}
    algos = ["lgbm", "catboost", "xgboost"]
    
    # 2. Her Algoritma için Çalışmaları Yürüt
    for algo in algos:
        logger.info(f"\n--- Optimizasyon Başlatılıyor: {algo} ---")
        
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        
        objective_func = make_composite_objective(algo, X, y, splits["folds"], sample_weight)
        
        # 30 trial (Apple Silicon performansı için optimal trial sayısı)
        study.optimize(objective_func, n_trials=30, show_progress_bar=True)
        
        logger.info(f"  En iyi trial skoru ({algo}): {study.best_value:.5f}")
        logger.info(f"  En iyi parametreler ({algo}): {study.best_params}")
        
        # Sonuçları kaydet
        joblib.dump(study, STUDIES_DIR / f"{algo}_study.joblib")
        best_params[algo] = study.best_params
        
    # 3. YENİDEN EĞİTİM (PIPELINE GÜNCELLEME)
    logger.info("\n=== AŞAMA 3: En İyi Parametrelerle Pipeline Güncelleniyor ===")
    
    quantiles = [0.1, 0.5, 0.9]
    oof_predictions = {}
    val_indices = None
    
    # Best parameter setleri ile Walk-Forward OOF üretimi
    for algo in algos:
        for q in quantiles:
            c_name = col_name(algo, q)
            logger.info(f"OOF tahmin matrisi üretiliyor (Optimum): {c_name}...")
            
            c_val_indices, c_oof_preds = make_oof_predictions(
                algo=algo,
                q=q,
                X=X,
                y=y,
                folds=splits["folds"],
                params=best_params[algo],
                sample_weight=sample_weight
            )
            
            if val_indices is None:
                val_indices = c_val_indices
            oof_predictions[c_name] = c_oof_preds
            
    x_meta_df = pd.DataFrame(oof_predictions, index=val_indices)
    
    # OOF meta matrisini ve validation indekslerini kaydet
    x_meta_data = {
        "val_indices": val_indices,
        "x_meta": x_meta_df,
        "y_meta": y.iloc[val_indices]
    }
    joblib.dump(x_meta_data, X_META_OUT)
    logger.info(f"Optimum OOF tahmin matrisi kaydedildi: {X_META_OUT}")
    
    # Fully-trained base modellerin best params ile eğitimi
    logger.info("\nFully-trained base modeller optimum parametrelerle eğitiliyor...")
    test_indices_set = set(splits["test_indices"])
    train_val_indices = np.array([i for i in range(len(df)) if i not in test_indices_set], dtype=np.int64)
    
    X_train_val = X.iloc[train_val_indices]
    y_train_val = y.iloc[train_val_indices]
    w_train_val = sample_weight.iloc[train_val_indices] if sample_weight is not None else None
    
    base_models = {}
    for algo in algos:
        for q in quantiles:
            c_name = col_name(algo, q)
            logger.info(f"Fully-trained model eğitiliyor (Optimum): {c_name}...")
            model = train_base_learner(
                algo=algo,
                q=q,
                X_train=X_train_val,
                y_train=y_train_val,
                params=best_params[algo],
                sample_weight=w_train_val
            )
            base_models[c_name] = model
            
    joblib.dump(base_models, BASE_MODELS_OUT)
    logger.info(f"Optimum fully-trained base modeller kaydedildi: {BASE_MODELS_OUT}")
    
    # 4. FASTQUANTILEREGRESSOR META-LEARNER EĞİTİMİ
    logger.info("\nPyTorch FastQuantileRegressor meta-öğrenici optimum girdilerle eğitiliyor...")
    
    flag_cols = ['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing']
    flags_val = df[flag_cols].iloc[val_indices].reset_index(drop=True)
    
    x_meta_reset = x_meta_df.reset_index(drop=True)
    x_meta_full = pd.concat([x_meta_reset, flags_val], axis=1)
    x_meta_full.index = val_indices
    
    meta_models = train_meta_learner(
        x_meta=x_meta_full,
        y=y.iloc[val_indices],
        quantiles=[0.1, 0.5, 0.9],
        max_iter=1500,
        lr=0.01
    )
    
    # OOF tahmin ve metrikler
    preds_oof = predict_intervals(meta_models, x_meta_full, enforce_monotonicity=True)
    
    # Gündüz/Üretim saatleri
    dl = df.iloc[val_indices]['cos_zenith'] > 0.087
    pos = y.iloc[val_indices] > 0
    daylight_mask = dl & pos
    
    picp_all = compute_coverage(y.iloc[val_indices], preds_oof)
    picp_daylight = compute_coverage(y.iloc[val_indices][daylight_mask], preds_oof[daylight_mask])
    
    logger.info(f"\n--- OOF Tahmin Metrikleri (Optimizasyon Sonrası) ---")
    logger.info(f"  OOF PICP (Tüm saatler - Gece dahil)   : {picp_all * 100:.2f}%")
    logger.info(f"  OOF PICP (Gündüz/Üretim Saatleri)     : {picp_daylight * 100:.2f}%  (Hedef ~80%)")
    
    payload = {
        "models": meta_models,
        "quantiles": [0.1, 0.5, 0.9],
        "columns": list(x_meta_full.columns),
        "oof_metrics": {
            "pinball_q10": pinball_loss(y.iloc[val_indices].values, preds_oof["q_0.1"].values, 0.1),
            "pinball_q50": pinball_loss(y.iloc[val_indices].values, preds_oof["q_0.5"].values, 0.5),
            "pinball_q90": pinball_loss(y.iloc[val_indices].values, preds_oof["q_0.9"].values, 0.9),
            "picp_all": picp_all,
            "picp_daylight": picp_daylight
        }
    }
    
    joblib.dump(payload, META_MODELS_OUT, compress=3)
    logger.info(f"Optimum meta modeller başarıyla kaydedildi: {META_MODELS_OUT}")
    logger.info("=== STAGE-7 TAMAMLANDI ===")

if __name__ == "__main__":
    main()
