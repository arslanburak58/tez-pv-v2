"""STAGE-4: Zaman serisi split doğrulama ve sızıntı engelleme testleri."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "features_v2.parquet"
SPLITS_FILE = PROCESSED_DIR / "splits.joblib"


def test_splits_file_exists() -> None:
    """splits.joblib dosyasının mevcut olduğunu doğrula."""
    assert SPLITS_FILE.exists(), "splits.joblib dosyası bulunamadı."


def test_splits_structure_and_types() -> None:
    """Splits nesnesinin beklenen anahtarlara ve veri tiplerine sahip olduğunu doğrula."""
    splits = joblib.load(SPLITS_FILE)
    
    assert isinstance(splits, dict)
    assert "test_indices" in splits
    assert "folds" in splits
    assert len(splits["folds"]) == 5
    
    assert isinstance(splits["test_indices"], np.ndarray)
    assert splits["test_indices"].dtype in [np.int64, np.int32]
    
    for fold in splits["folds"]:
        assert "train_indices" in fold
        assert "val_indices" in fold
        assert isinstance(fold["train_indices"], np.ndarray)
        assert isinstance(fold["val_indices"], np.ndarray)


def test_zero_leakage_and_intersections() -> None:
    """Train, Val ve Test indekslerinin tamamen ayrık (disjoint) olduğunu doğrula."""
    splits = joblib.load(SPLITS_FILE)
    test_set = set(splits["test_indices"])
    
    # Test seti boyutu makul olmalı
    assert len(test_set) > 0, "Test seti indeksleri boş."
    
    for fold_idx, fold in enumerate(splits["folds"]):
        train_set = set(fold["train_indices"])
        val_set = set(fold["val_indices"])
        
        # Train ve Val kesişimi boş küme olmalı
        train_val_intersection = train_set.intersection(val_set)
        assert len(train_val_intersection) == 0, f"Fold {fold_idx}: Train ve Val indeksleri çakışıyor! Çakışan adet: {len(train_val_intersection)}"
        
        # Test seti hiçbir fold'un train veya val setinde yer almamalıdır
        test_train_intersection = test_set.intersection(train_set)
        assert len(test_train_intersection) == 0, f"Fold {fold_idx}: Test indeksleri Train setine sızmış! Adet: {len(test_train_intersection)}"
        
        test_val_intersection = test_set.intersection(val_set)
        assert len(test_val_intersection) == 0, f"Fold {fold_idx}: Test indeksleri Validation setine sızmış! Adet: {len(test_val_intersection)}"


def test_time_series_gap() -> None:
    """Her fold ve her istasyon için train sonu ile val başlangıcı arasında en az 24 saatlik boşluk (gap) olduğunu doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    df = df.reset_index(drop=True)
    df["original_index"] = np.arange(len(df))
    
    splits = joblib.load(SPLITS_FILE)
    station_ids = df["station_id"].unique()
    
    for fold_idx, fold in enumerate(splits["folds"]):
        train_indices_set = set(fold["train_indices"])
        val_indices_set = set(fold["val_indices"])
        
        for st_id in station_ids:
            st_df = df[df["station_id"] == st_id]
            st_indices = set(st_df["original_index"].values)
            
            # Bu istasyonun bu fold'daki train ve val indekslerini bul
            st_train_idxs = list(st_indices.intersection(train_indices_set))
            st_val_idxs = list(st_indices.intersection(val_indices_set))
            
            # Eğer ikisi de doluysa, aradaki zaman farkını kontrol et
            if len(st_train_idxs) > 0 and len(st_val_idxs) > 0:
                st_train_df = st_df[st_df["original_index"].isin(st_train_idxs)]
                st_val_df = st_df[st_df["original_index"].isin(st_val_idxs)]
                
                max_train_time = st_train_df["timestamp"].max()
                min_val_time = st_val_df["timestamp"].min()
                
                # Sadece zaman sıralı olup olmadığını doğrula
                assert min_val_time > max_train_time, f"Fold {fold_idx}, İstasyon {st_id}: Kronolojik sıra ihlali! Val başlangıcı: {min_val_time}, Train bitişi: {max_train_time}"
                
                # Gap kontrolü (en az 24 saat olmalı)
                gap_duration = min_val_time - max_train_time
                assert gap_duration >= pd.Timedelta(hours=24), (
                    f"Fold {fold_idx}, İstasyon {st_id}: Gap süresi 24 saatten az! "
                    f"Mevcut gap: {gap_duration}, Train Bitiş: {max_train_time}, Val Başlangıç: {min_val_time}"
                )


def test_holdout_isolation() -> None:
    """Holdout istasyonlarının eğitim ve test indekslerinde kesinlikle yer almadığını doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    df = df.reset_index(drop=True)
    
    splits = joblib.load(SPLITS_FILE)
    
    # splits.joblib içindeki tüm indeksleri topla
    all_split_indices = list(splits["test_indices"])
    for fold in splits["folds"]:
        all_split_indices.extend(fold["train_indices"])
        all_split_indices.extend(fold["val_indices"])
        
    all_split_indices = np.array(all_split_indices)
    
    # Bu indekslerin işaret ettiği satırlardaki istasyonları al
    active_stations = df.loc[all_split_indices, "station_id"].unique()
    
    assert "station02" not in active_stations, "Hata: Holdout istasyonu station02 split indeksleri içinde yer alıyor!"
    assert "station09" not in active_stations, "Hata: Holdout istasyonu station09 split indeksleri içinde yer alıyor!"
