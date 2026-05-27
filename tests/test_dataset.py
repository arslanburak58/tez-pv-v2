"""STAGE-2: Veri seti doğrulama testleri."""

from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
HOLDOUT_DIR = PROCESSED_DIR / "holdout"


def test_processed_files_exist() -> None:
    """Veri seti ve holdout parquet dosyalarının mevcut olmasını doğrula."""
    assert (PROCESSED_DIR / "dataset_v2.parquet").exists(), "dataset_v2.parquet bulunamadı."
    assert (HOLDOUT_DIR / "station02.parquet").exists(), "station02.parquet bulunamadı."
    assert (HOLDOUT_DIR / "station09.parquet").exists(), "station09.parquet bulunamadı."


def test_combined_dataset_structure() -> None:
    """Eğitim veri kümesi sütunlarını, tiplerini ve null değer içermediğini doğrula."""
    df = pd.read_parquet(PROCESSED_DIR / "dataset_v2.parquet")
    
    expected_cols = {
        "timestamp", "station_id", "lat", "lon", "capacity_kW", "power_kW", "y_norm",
        "GHI", "T_amb", "RH", "GHI_is_missing", "T_amb_is_missing", "RH_is_missing"
    }
    
    # Sütun isimlerinin tam eşleştiğini kontrol et
    assert set(df.columns) == expected_cols, f"Sütunlar eşleşmiyor. Mevcut: {df.columns.tolist()}"
    
    # Rüzgar hızı sütunlarının OLMADIĞINI doğrula
    assert "WS" not in df.columns, "Rüzgar hızı (WS) sütunu bulunmamalıdır."
    assert "WS_is_missing" not in df.columns, "WS_is_missing bayrağı bulunmamalıdır."
    
    # Hiçbir satırda null kalmadığını kontrol et
    nulls = df.isnull().sum()
    for col in df.columns:
        assert nulls[col] == 0, f"{col} sütununda {nulls[col]} adet null değer bulundu."
        
    # Veri tiplerinin doğruluğunu test et
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp sütunu datetime tipinde olmalıdır."
    assert df["timestamp"].dt.tz is not None, "timestamp sütunu timezone-aware (tz-aware) olmalıdır."
    assert df["timestamp"].dt.tz.zone == "UTC", "timestamp sütunu UTC zaman diliminde olmalıdır."
    
    # İstasyonları doğrula (station02 ve station09 kesinlikle bulunmamalı)
    station_ids = df["station_id"].unique()
    assert "dkasc_alice_springs" in station_ids
    assert "station00" in station_ids
    assert "station02" not in station_ids, "Holdout istasyonu station02 eğitim kümesinde bulunmamalıdır."
    assert "station09" not in station_ids, "Holdout istasyonu station09 eğitim kümesinde bulunmamalıdır."
    assert len(station_ids) == 9, f"Toplam 9 istasyon olmalı, fakat {len(station_ids)} bulundu."


def test_y_norm_bounds() -> None:
    """y_norm değerlerinin [0.0, 1.5] aralığında olmasını doğrula."""
    df = pd.read_parquet(PROCESSED_DIR / "dataset_v2.parquet")
    
    assert (df["y_norm"] >= 0.0).all(), "y_norm negatif değerler içeremez."
    assert (df["y_norm"] <= 1.5).all(), "y_norm 1.5'ten büyük değerler içeremez (spike toleransı dahil)."


def test_holdout_datasets() -> None:
    """Holdout istasyonlarının verilerini ve yapısal doğruluğunu kontrol et."""
    for st_id in ["station02", "station09"]:
        f_path = HOLDOUT_DIR / f"{st_id}.parquet"
        df = pd.read_parquet(f_path)
        
        # Sütunları kontrol et
        expected_cols = {
            "timestamp", "station_id", "lat", "lon", "capacity_kW", "power_kW", "y_norm",
            "GHI", "T_amb", "RH", "GHI_is_missing", "T_amb_is_missing", "RH_is_missing"
        }
        assert set(df.columns) == expected_cols
        assert "WS" not in df.columns
        
        # İstasyon ID'sini doğrula
        assert (df["station_id"] == st_id).all(), f"Verideki istasyon ID'si {st_id} ile uyuşmuyor."
        
        # Null ve sınır kontrolleri
        assert df.isnull().sum().sum() == 0, f"{st_id} içinde null değerler bulundu."
        assert (df["y_norm"] >= 0.0).all()
        assert (df["y_norm"] <= 1.5).all()
        
        # Zaman damgasını kontrol et
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].dt.tz is not None
        assert df["timestamp"].dt.tz.zone == "UTC"
