"""STAGE-4: Zaman serisi çapraz doğrulama (cross-validation) ve test bölünmelerini oluşturan betik."""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Günlükleme ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FEATURES_FILE = ROOT / "data" / "processed" / "features_v2.parquet"
SPLITS_OUT_FILE = ROOT / "data" / "processed" / "splits.joblib"


def main() -> None:
    logger.info("Zaman serisi bölünme işlemleri başlatılıyor...")
    
    # 1. Veri kümesini yükle
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} bulunamadı. Lütfen önce STAGE-3'ü tamamlayın.")
        
    df = pd.read_parquet(FEATURES_FILE)
    logger.info(f"Birleşik öznitelik verisi yüklendi: {df.shape}")
    
    # Holdout istasyonlarının eğitim kümesinde bulunmadığını doğrula
    station_ids = df["station_id"].unique()
    assert "station02" not in station_ids, "Hata: Holdout istasyonu station02 eğitim kümesinde yer alıyor!"
    assert "station09" not in station_ids, "Hata: Holdout istasyonu station09 eğitim kümesinde yer alıyor!"
    
    # Orijinal satır indekslerini kaybetmemek için sütun ekle
    df = df.reset_index(drop=True)
    df["original_index"] = np.arange(len(df))
    
    # Bölünme yapısı hazırlığı
    splits = {
        "test_indices": [],
        "folds": [
            {"train_indices": [], "val_indices": []}
            for _ in range(5)
        ]
    }
    
    # Her istasyon için bağımsız zaman serisi bölünmesi
    for st_id in station_ids:
        st_df = df[df["station_id"] == st_id].copy()
        
        # Zaman damgasına göre kronolojik sırala
        st_df = st_df.sort_values("timestamp")
        st_indices = st_df["original_index"].values
        st_times = st_df["timestamp"].values
        
        n_samples = len(st_df)
        n_test = int(n_samples * 0.2)
        n_train_val = n_samples - n_test
        
        # Test seti (%20)
        st_test_indices = st_indices[n_train_val:]
        splits["test_indices"].extend(st_test_indices)
        
        # Train/Val seti (%80)
        st_train_val_indices = st_indices[:n_train_val]
        
        # 24 saatlik boşluğu örneklem sıklığına göre dinamik olarak belirle
        # DKASC: 5-dakikalık (288 satır/gün) | PVOD: 15-dakikalık (96 satır/gün)
        st_gap = 288 if st_id == "dkasc_alice_springs" else 96
        
        # 5-Fold TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=5, gap=st_gap)
        
        logger.info(f"İstasyon: {st_id:25} | Toplam Satır: {n_samples:8} | Test Satır: {n_test:6} | Train/Val Satır: {n_train_val:6}")
        
        # Her fold için bölünme
        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(st_train_val_indices)):
            st_train_indices = st_train_val_indices[train_idx]
            st_val_indices = st_train_val_indices[val_idx]
            
            splits["folds"][fold_idx]["train_indices"].extend(st_train_indices)
            splits["folds"][fold_idx]["val_indices"].extend(st_val_indices)
            
            # Zaman aralıklarını logla (sadece ilk ve son elemanlar üzerinden)
            train_start = st_times[train_idx[0]]
            train_end = st_times[train_idx[-1]]
            val_start = st_times[val_idx[0]]
            val_end = st_times[val_idx[-1]]
            logger.debug(
                f"  Fold {fold_idx}: Train [{train_start} - {train_end}] ({len(st_train_indices)} satır) | "
                f"Val [{val_start} - {val_end}] ({len(st_val_indices)} satır)"
            )
            
    # Listeleri numpy array'lerine dönüştür ve doğrula
    splits["test_indices"] = np.array(splits["test_indices"], dtype=np.int64)
    logger.info(f"Global Test Seti Toplam Satır Sayısı: {len(splits['test_indices'])}")
    
    for fold_idx in range(5):
        splits["folds"][fold_idx]["train_indices"] = np.array(splits["folds"][fold_idx]["train_indices"], dtype=np.int64)
        splits["folds"][fold_idx]["val_indices"] = np.array(splits["folds"][fold_idx]["val_indices"], dtype=np.int64)
        
        logger.info(
            f"Global Fold {fold_idx}: "
            f"Train = {len(splits['folds'][fold_idx]['train_indices']):8} satır | "
            f"Val = {len(splits['folds'][fold_idx]['val_indices']):8} satır"
        )
        
    # Splits klasörünün varlığından emin ol
    SPLITS_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # joblib ile kaydet
    joblib.dump(splits, SPLITS_OUT_FILE)
    logger.info(f"Bölünme indeksleri başarıyla kaydedildi: {SPLITS_OUT_FILE}")
    logger.info("STAGE-4 Zaman serisi bölünme işlemleri tamamlandı!")


if __name__ == "__main__":
    main()
