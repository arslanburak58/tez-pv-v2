"""STAGE-3: Fiziksel öznitelik testleri."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "features_v2.parquet"
HOLDOUT_DIR = PROCESSED_DIR / "holdout"


def test_features_file_exists() -> None:
    """Öznitelik dosyalarının mevcut olmasını doğrula."""
    assert FEATURES_FILE.exists(), "features_v2.parquet bulunamadı."
    assert (HOLDOUT_DIR / "station02.parquet").exists()
    assert (HOLDOUT_DIR / "station09.parquet").exists()


def test_feature_columns_and_nans() -> None:
    """Sütun yapısını ve null değerlerin olmadığını doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    
    expected_cols = {
        "timestamp", "station_id", "lat", "lon", "capacity_kW", "power_kW", "y_norm",
        "GHI", "T_amb", "RH", "GHI_is_missing", "T_amb_is_missing", "RH_is_missing",
        "apparent_zenith", "zenith", "apparent_elevation", "elevation", "azimuth",
        "cos_zenith", "airmass", "hour_angle", "k_t", "T_cell",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    }
    
    assert set(df.columns) == expected_cols, f"Sütun hatası. Eksik veya fazla: {set(df.columns) ^ expected_cols}"
    assert df.isnull().sum().sum() == 0, "Öznitelik matrisinde NaN değerler bulundu."


def test_cos_zenith_ranges() -> None:
    """cos_zenith aralığını ve gece/gündüz sınırlarını doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    
    cos_zenith = df["cos_zenith"]
    assert (cos_zenith >= -1.0).all() and (cos_zenith <= 1.0).all(), "cos_zenith [-1, 1] dışında."
    
    # Zenith > 90 derece ise gece olmalı ve cos_zenith < 0 olmalı
    gece_mask = df["zenith"] > 90.0
    assert (cos_zenith[gece_mask] < 0.0).all(), "Gece saatlerinde cos_zenith < 0 olmalıdır."
    
    # Zenith < 90 derece ise gündüz olmalı ve cos_zenith > 0 olmalı
    gunduz_mask = df["zenith"] < 90.0
    assert (cos_zenith[gunduz_mask] > 0.0).all(), "Gündüz saatlerinde cos_zenith > 0 olmalıdır."


def test_k_t_physical_ranges() -> None:
    """Clear-Sky Index (k_t) sınır kontrollerini doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    
    k_t = df["k_t"]
    assert (k_t >= 0.0).all() and (k_t <= 2.0).all(), "k_t [0.0, 2.0] sınırları dışında."
    
    # Eşik Kontrolü: Zenith >= 85 derece ise k_t == 0.0 olmalı
    gece_sinir_mask = df["zenith"] >= 85.0
    assert (k_t[gece_sinir_mask] == 0.0).all(), "Zenit >= 85° iken k_t sıfır olmalıdır."
    
    # Gündüz saatlerindeki k_t ortalaması makul olmalı [0.3, 0.8]
    gunduz_kt = k_t[df["zenith"] < 80.0]
    mean_gunduz_kt = gunduz_kt.mean()
    assert 0.3 <= mean_gunduz_kt <= 0.8, f"Ortalama gündüz k_t ({mean_gunduz_kt:.2f}) kabul edilemez düzeyde."


def test_cell_temp_physical() -> None:
    """Panel Sıcaklığının (T_cell) termal fizik kurallarına uygunluğunu doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    
    # T_cell >= T_amb olmalı (çünkü GHI >= 0 ve Ross katsayısı > 0)
    assert (df["T_cell"] >= df["T_amb"]).all(), "T_cell, T_amb'den küçük olamaz."
    
    # Yüksek ışınım altında panel ısınması testi (Ross Doyum Testi)
    # GHI > 800 W/m² iken T_cell - T_amb farkı ortalama ~15-35 °C arasında olmalıdır
    gunesli_mask = df["GHI"] > 800.0
    temp_diff = df.loc[gunesli_mask, "T_cell"] - df.loc[gunesli_mask, "T_amb"]
    
    mean_diff = temp_diff.mean()
    assert 15.0 <= mean_diff <= 35.0, f"Yüksek ışınımda panel ısınma farkı ({mean_diff:.2f}°C) fiziksel olarak mantıksız."


def test_cyclical_bounds() -> None:
    """Döngüsel zaman değişkenlerinin matematiksel sınırlarını doğrula."""
    df = pd.read_parquet(FEATURES_FILE)
    
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert (df[col] >= -1.0001).all() and (df[col] <= 1.0001).all(), f"{col} döngüsel sınırlar dışında."
        
    # Trigonometrik özdeşlik: sin^2 + cos^2 = 1.0
    hour_identity = df["hour_sin"]**2 + df["hour_cos"]**2
    assert np.allclose(hour_identity, 1.0, atol=1e-5), "hour sin/cos trigonometrik özdeşliği sağlamıyor."
