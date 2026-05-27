"""STAGE-5: Base learner eğitim ve çıktı doğrulama testleri."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
BASE_MODELS_FILE = PROCESSED_DIR / "base_models_v2.joblib"
X_META_FILE = PROCESSED_DIR / "x_meta_v2.joblib"


def test_files_exist() -> None:
    """Eğitilen modellerin ve OOF matrislerinin varlığını doğrula."""
    assert BASE_MODELS_FILE.exists(), "base_models_v2.joblib dosyası bulunamadı."
    assert X_META_FILE.exists(), "x_meta_v2.joblib dosyası bulunamadı."


def test_base_models_dictionary() -> None:
    """Modeller sözlüğünün yapısını ve tüm 9 modelin eğitildiğini doğrula."""
    models = joblib.load(BASE_MODELS_FILE)
    
    assert isinstance(models, dict)
    assert len(models) == 9
    
    expected_keys = {
        "lgbm_q01", "lgbm_q05", "lgbm_q09",
        "catboost_q01", "catboost_q05", "catboost_q09",
        "xgboost_q01", "xgboost_q05", "xgboost_q09"
    }
    assert set(models.keys()) == expected_keys


def test_x_meta_structure_and_nans() -> None:
    """OOF tahmin matrisinin yapısını, satır sayılarını ve null içermediğini doğrula."""
    data = joblib.load(X_META_FILE)
    
    assert isinstance(data, dict)
    assert "val_indices" in data
    assert "x_meta" in data
    assert "y_meta" in data
    
    x_meta = data["x_meta"]
    y_meta = data["y_meta"]
    val_indices = data["val_indices"]
    
    # Satır sayısı kontrolü (5 fold'un toplam validation satır sayısı: 1,051,950)
    assert len(x_meta) == 1051950, f"Beklenen 1,051,950 satır, fakat fiili satır: {len(x_meta)}"
    assert len(y_meta) == 1051950
    assert len(val_indices) == 1051950
    
    # Sütun isimleri kontrolü
    expected_cols = {
        "lgbm_q01", "lgbm_q05", "lgbm_q09",
        "catboost_q01", "catboost_q05", "catboost_q09",
        "xgboost_q01", "xgboost_q05", "xgboost_q09"
    }
    assert set(x_meta.columns) == expected_cols
    
    # Hiçbir sütunda null kalmadığını kontrol et
    assert x_meta.isnull().sum().sum() == 0, "OOF tahmin matrisinde NaN/Null değerler mevcut!"
    
    # Sınırlar kontrolü (tahminler makul y_norm sınırlarında olmalı, örn. [-0.5, 2.0] arası)
    assert (x_meta >= -0.5).all().all(), "Tahminlerde çok yüksek negatif sızıntı var (<-0.5)."
    assert (x_meta <= 2.0).all().all(), "Tahminlerde aşırı yüksek spikes var (>2.0)."


def test_quantile_monotonicity() -> None:
    """OOF tahminlerinde her algoritma için q01 <= q05 <= q09 monotonluğunun ortalama düzeyde sağlandığını doğrula."""
    data = joblib.load(X_META_FILE)
    x_meta = data["x_meta"]
    
    # Gündüz/Gece ayrımı için GHI değerlerini yükle
    df = pd.read_parquet(PROCESSED_DIR / "features_v2.parquet")
    ghi_val = df.loc[x_meta.index, "GHI"]
    daytime_mask = ghi_val > 50.0
    
    algos = ["lgbm", "catboost", "xgboost"]
    
    for algo in algos:
        col_01 = f"{algo}_q01"
        col_05 = f"{algo}_q05"
        col_09 = f"{algo}_q09"
        
        # Ortalama düzeyde quantile monotonluk testi (tüm veri setinde)
        mean_01 = x_meta[col_01].mean()
        mean_05 = x_meta[col_05].mean()
        mean_09 = x_meta[col_09].mean()
        
        assert mean_01 < mean_05, f"{algo} için ortalama q01 >= q05! ({mean_01:.4f} >= {mean_05:.4f})"
        assert mean_05 < mean_09, f"{algo} için ortalama q05 >= q09! ({mean_05:.4f} >= {mean_09:.4f})"
        
        # Noktasal düzeyde crossing oranı gündüz saatlerinde %15'ten az olmalı (fiilen ~%1 civarındadır)
        # Geceleri ışınım sıfır iken tüm quantiles 0 etrafında dalgalandığından matematiksel crossing oranı yanıltıcıdır
        crossing_01_05 = (x_meta.loc[daytime_mask, col_01] > x_meta.loc[daytime_mask, col_05]).mean()
        crossing_05_09 = (x_meta.loc[daytime_mask, col_05] > x_meta.loc[daytime_mask, col_09]).mean()
        
        assert crossing_01_05 < 0.15, f"{algo} için gündüz q01 > q05 crossing oranı çok yüksek: {crossing_01_05:.2%}"
        assert crossing_05_09 < 0.15, f"{algo} için gündüz q05 > q09 crossing oranı çok yüksek: {crossing_05_09:.2%}"
