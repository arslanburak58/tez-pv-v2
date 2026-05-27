"""STAGE-5: 9 base modelin (XGBoost, LightGBM, CatBoost × 3 quantile) eğitilmesi ve OOF tahmin matrisinin üretilmesi."""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from models.base_learners import make_oof_predictions, train_base_learner, col_name, pinball_loss

# Günlükleme ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FEATURES_FILE = ROOT / "data" / "processed" / "features_v2.parquet"
SPLITS_FILE = ROOT / "data" / "processed" / "splits.joblib"
BASE_MODELS_OUT = ROOT / "data" / "processed" / "base_models_v2.joblib"
X_META_OUT = ROOT / "data" / "processed" / "x_meta_v2.joblib"


def main() -> None:
    logger.info("Base learner eğitim süreci başlatılıyor...")
    
    # 1. Girdileri yükle
    if not FEATURES_FILE.exists() or not SPLITS_FILE.exists():
        raise FileNotFoundError("Gerekli girdi dosyaları bulunamadı. Lütfen STAGE-3 ve STAGE-4'ü tamamlayın.")
        
    df = pd.read_parquet(FEATURES_FILE)
    splits = joblib.load(SPLITS_FILE)
    
    logger.info(f"Öznitelik matrisi yüklendi: {df.shape}")
    logger.info(f"Bölünme indeksleri yüklendi.")
    
    # 2. X ve y ayrımını yap
    # Eğitim dışı sütunları ayır
    exclude_cols = ["timestamp", "station_id", "power_kW", "y_norm", "original_index"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Model eğitiminde kullanılacak {len(feature_cols)} öznitelik belirlendi:")
    logger.info(f"  {feature_cols}")
    
    X = df[feature_cols].copy()
    y = df["y_norm"].copy()
    
    quantiles = [0.1, 0.5, 0.9]
    algos = ["lgbm", "catboost", "xgboost"]
    
    # --- 3. OUT-OF-FOLD (OOF) TAHMİNLERİNİN ÜRETİLMESİ ---
    logger.info("\n=== AŞAMA 1: Walk-Forward OOF Tahminlerinin Üretilmesi ===")
    
    oof_predictions = {}
    val_indices = None
    
    for algo in algos:
        for q in quantiles:
            c_name = col_name(algo, q)
            logger.info(f"OOF tahminler üretiliyor: {c_name}...")
            
            # Walk-forward OOF tahminleri üret
            c_val_indices, c_oof_preds = make_oof_predictions(
                algo=algo,
                q=q,
                X=X,
                y=y,
                folds=splits["folds"]
            )
            
            # Doğrulama indekslerinin tüm modeller için aynı sırayla döndüğünü teyit et
            if val_indices is None:
                val_indices = c_val_indices
            else:
                assert np.array_equal(val_indices, c_val_indices), "Hata: Modeller arasında validation indeks sırası uyuşmuyor!"
                
            oof_predictions[c_name] = c_oof_preds
            
            # Ortalama OOF Pinball loss değerini hesapla ve logla
            y_val_actual = y.iloc[val_indices].values
            mean_loss = pinball_loss(y_val_actual, c_oof_preds, q)
            logger.info(f"  -> Model {c_name} Ortalama OOF Pinball Loss: {mean_loss:.5f}")
            
    # OOF veri çerçevesini (DataFrame) oluştur
    x_meta_df = pd.DataFrame(oof_predictions, index=val_indices)
    
    x_meta_data = {
        "val_indices": val_indices,
        "x_meta": x_meta_df,
        "y_meta": y.iloc[val_indices]
    }
    
    # x_meta_v2.joblib dosyasını kaydet
    joblib.dump(x_meta_data, X_META_OUT)
    logger.info(f"OOF tahmin matrisi kaydedildi: {X_META_OUT} (Boyut: {x_meta_df.shape})")
    
    # --- 4. TÜM EĞİTİM KÜMESİ ÜZERİNDE FULLY-TRAINED MODELLERİN EĞİTİLMESİ ---
    logger.info("\n=== AŞAMA 2: Tüm Eğitim Kümesi Üzerinde Fully-Trained Modellerin Eğitilmesi ===")
    
    # Eğitim + Validation kümesi = Test setine girmeyen tüm indeksler
    test_indices_set = set(splits["test_indices"])
    train_val_indices = np.array([i for i in range(len(df)) if i not in test_indices_set], dtype=np.int64)
    
    X_train_val = X.iloc[train_val_indices]
    y_train_val = y.iloc[train_val_indices]
    
    logger.info(f"Fully-trained modeller {len(train_val_indices)} satırlık eğitim+validation kümesinde eğitilecek.")
    
    base_models = {}
    
    for algo in algos:
        for q in quantiles:
            c_name = col_name(algo, q)
            logger.info(f"Fully-trained model eğitiliyor: {c_name}...")
            
            model = train_base_learner(
                algo=algo,
                q=q,
                X_train=X_train_val,
                y_train=y_train_val
            )
            
            base_models[c_name] = model
            logger.info(f"  -> Model {c_name} başarıyla eğitildi.")
            
    # base_models_v2.joblib dosyasını kaydet
    joblib.dump(base_models, BASE_MODELS_OUT)
    logger.info(f"Tüm fully-trained base modeller başarıyla kaydedildi: {BASE_MODELS_OUT}")
    
    logger.info("\nSTAGE-5 Base learner eğitimi başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
