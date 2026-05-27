# SKILL: Python Kodu Yardımı

## Ne zaman bu skill aktive olmalı
Python kodu, debug, pipeline tasarımı, kütüphane kullanımı soruları.

## Standartlar

**Python sürümü:** 3.13.x (güncellendi - Anaconda base)
- joblib uyumluluğu için sürümü bozma
- .python-version dosyası kullan gerekirse

**Kod stili:**
- PEP 8 + ruff
- Tip ipuçları **zorunlu**
- NumPy stili docstring
- f-string > %-formatting

**Reprodüksiyon:**
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
```

**Donanım:**
- MacBook Air M4 (Apple Silicon)
- PyTorch için MPS backend: `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`
- Bellek sınırı M4'te dar — LSTM/Transformer'da Colab T4 öner
- Tree-based modeller (XGB/LGBM/CatBoost) M4'te rahat

**Veri yapıları:**
- pandas DataFrame + DatetimeIndex (tz-aware)
- Büyük veri için parquet (pyarrow)
- Model serialization: joblib

**Pipeline kuralları:**
- Scaler/imputer/encoder SADECE train'de fit, val/test'te transform
- TimeSeriesSplit gap=24*4 (96 örnek, 24 saat)
- OOF predictions ile meta-input oluştur

## v2 Spesifik Kurallar

**Hedef değişken:** `y_norm = power_kW / capacity_kW ∈ [0,1]`
Asla mutlak kW ile model eğitme. Çıkış mutlak istenirse:
```python
y_kW = y_norm_pred * capacity_kW
```

**Meta-öğrenici:** sklearn QuantileRegressor
```python
from sklearn.linear_model import QuantileRegressor
qr = QuantileRegressor(quantile=0.1, solver='highs', alpha=0.0)
```

**Quantile crossing kontrolü:** Post-prediction sort
```python
preds_sorted = np.sort(preds.values, axis=1)
```

## Yapma
- `from X import *`
- Magic number (sabitler config'de)
- Mutlak kW ile training
- CQR post-processing (LP gerek kalmasın)
- Pickle (joblib kullan)
- Veri sızıntısı (test'te fit)
- StringDtype uyumsuzlukları
