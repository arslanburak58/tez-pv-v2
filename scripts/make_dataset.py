"""
STAGE-2: Veri ETL ve Kapasite Normalizasyonu

Bu script:
1. DKASC + PVOD ham verilerini okur
2. PVOD power birimini doğrular ve kW'a normalize eder
3. Her satır için y_norm = power_kW / capacity_kW hesaplar
4. Holdout istasyonlarını ayırır (station02, station09)
5. Birleşik DataFrame'i parquet'e kaydeder

Çıktı:
- data/processed/dataset_v2.parquet (training data)
- data/processed/holdout/station02.parquet
- data/processed/holdout/station09.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
HOLDOUT_DIR = OUT_DIR / "holdout"
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_STATIONS = ["station02", "station09"]


def verify_pvod_power_unit(df: pd.DataFrame, capacity_kW: float) -> str:
    """PVOD power'ın kW mı MW mı olduğunu tespit et.

    Hipotez: kW birimde olmalı. Test:
    - capacity_factor = power.max() / capacity_kW
    - Eğer 0-1 arası → kW (doğru)
    - Eğer 0.001 civarı → MW (×1000 düzeltmesi gerekli)
    """
    cf = df["power"].max() / capacity_kW
    if 0.05 < cf < 1.05:
        return "kW"
    elif 0.00005 < cf < 0.00105:
        return "MW"
    else:
        raise ValueError(
            f"Power birimini tespit edilemedi. "
            f"max={df['power'].max()}, capacity={capacity_kW}, cf={cf}"
        )


def main() -> None:
    raise NotImplementedError("STAGE-2 başladığında tam implementasyon")


if __name__ == "__main__":
    main()
