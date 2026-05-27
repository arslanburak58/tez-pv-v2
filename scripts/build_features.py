"""
STAGE-3: Fiziksel Öznitelik Mühendisliği

Bu script:
1. data/processed/dataset_v2.parquet ve holdout istasyon parquet dosyalarını yükler.
2. pvlib kullanarak her istasyon için solar position, cos_zenith ve airmass hesaplar.
3. Yerel Güneş Saati (LST) üzerinden Saat Açısını (H) hesaplar.
4. Berraklık İndeksini (k_t = GHI / G_extra_horizontal) eşik limitleriyle hesaplar.
5. Ross Formülü ile panel sıcaklığını (T_cell) hesaplar.
6. İstasyonların kendi YEREL saat dilimlerine göre döngüsel (cyclical) hour_sin, hour_cos, month_sin, month_cos özniteliklerini hesaplar.
7. Öznitelik eklenmiş veri setlerini parquet olarak kaydeder.

Çıktı:
- data/processed/features_v2.parquet (Eğitim veri kümesi öznitelikleri)
- data/processed/holdout/station02.parquet (Holdout 02 öznitelikleri güncellendi)
- data/processed/holdout/station09.parquet (Holdout 09 öznitelikleri güncellendi)
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm

# Reprodüksiyon için tohumlar
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
HOLDOUT_DIR = PROCESSED_DIR / "holdout"


def calculate_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tek bir istasyon için pvlib astronomik ve termal özniteliklerini hesaplar.

    Parameters
    ----------
    df : pd.DataFrame
        Tek bir istasyonun veri çerçevesi (sadece o istasyona ait).

    Returns
    -------
    pd.DataFrame
        Fiziksel öznitelikler eklenmiş veri çerçevesi.
    """
    df = df.copy()
    
    st_id = df["station_id"].iloc[0]
    lat = float(df["lat"].iloc[0])
    lon = float(df["lon"].iloc[0])
    
    # 1. DatetimeIndex ve Saat Dilimleri
    times = pd.DatetimeIndex(df["timestamp"])
    
    # İstasyonun yerel standard saat dilimi
    tz = "Australia/Darwin" if st_id == "dkasc_alice_springs" else "Asia/Shanghai"
    times_local = times.tz_convert(tz)
    
    # 2. Solar Position Hesaplama
    solpos = pvlib.solarposition.get_solarposition(times, latitude=lat, longitude=lon)
    df["apparent_zenith"] = solpos["apparent_zenith"].values
    df["zenith"] = solpos["zenith"].values
    df["apparent_elevation"] = solpos["apparent_elevation"].values
    df["elevation"] = solpos["elevation"].values
    df["azimuth"] = solpos["azimuth"].values
    
    # 3. cos_zenith
    df["cos_zenith"] = np.cos(np.radians(df["zenith"].values))
    
    # 4. Bağıl Airmass (Relative Airmass)
    airmass = pvlib.atmosphere.get_relative_airmass(df["apparent_zenith"].values)
    # Gece saatlerinde airmass'ı 0 yap, gündüzü 20 ile clip et
    df["airmass"] = np.where(df["zenith"].values < 90.0, np.clip(airmass, 0.0, 20.0), 0.0)
    
    # 5. Saat Açısı (Hour Angle - H)
    # UTC saatini kesirli biçimde al
    times_utc = times.tz_convert("UTC")
    utc_hour = times_utc.hour + times_utc.minute / 60.0 + times_utc.second / 3600.0
    
    # Yerel Güneş Saati (Local Solar Time - LST)
    eot = solpos["equation_of_time"].values
    lst_hours = (utc_hour + lon / 15.0 + eot / 60.0) % 24
    
    # Saat Açısı H = (LST - 12) * 15
    hour_angle = (lst_hours - 12.0) * 15.0
    # [-180, 180] aralığına sığdır (Wrap)
    df["hour_angle"] = (hour_angle + 180.0) % 360.0 - 180.0
    
    # 6. Berraklık İndeksi (Clear-Sky Index - k_t)
    # Dünya dışı yatay ışınım (G_extra_horizontal)
    ghi_extra = pvlib.irradiance.get_extra_radiation(times).values
    ghi_extra_horizontal = ghi_extra * df["cos_zenith"].values
    
    # Eşik Kontrolü: zenith < 85° gündüz ise k_t = GHI / G_extra_horizontal, aksi halde 0.0
    kt = np.where(
        df["zenith"].values < 85.0,
        df["GHI"].values / np.maximum(ghi_extra_horizontal, 1.0),
        0.0
    )
    df["k_t"] = np.clip(kt, 0.0, 2.0)
    
    # 7. Ross Formülü ile Panel Sıcaklığı (T_cell)
    # T_cell = T_amb + k * GHI  (NOCT = 45°C için k = (45 - 20) / 800 = 0.03125)
    df["T_cell"] = df["T_amb"].values + 0.03125 * df["GHI"].values
    
    # 8. Döngüsel Zaman Öznitelikleri (İstasyonun kendi YEREL saatine göre hizalı)
    local_hour = times_local.hour + times_local.minute / 60.0
    local_month = times_local.month - 1  # 0-indexed (0 = Ocak)
    
    df["hour_sin"] = np.sin(2 * np.pi * local_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * local_hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * local_month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * local_month / 12.0)
    
    return df


def main() -> None:
    """Veri setleri üzerinde STAGE-3 öznitelik mühendisliğini çalıştırır."""
    # 1. Birleşik eğitim veri setini oku
    train_parquet = PROCESSED_DIR / "dataset_v2.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError("dataset_v2.parquet bulunamadı. Lütfen önce STAGE-2'yi koşturun.")
        
    print("dataset_v2.parquet yükleniyor...")
    df_train = pd.read_parquet(train_parquet)
    
    # İstasyon bazında gruplayıp öznitelikleri paralel/sırayla hesaplayalım
    print("Eğitim istasyonlarının fiziksel öznitelikleri hesaplanıyor...")
    station_groups = df_train.groupby("station_id")
    
    processed_dfs = []
    for st_id, group_df in tqdm(station_groups, desc="Eğitim Grupları"):
        processed_df = calculate_station_features(group_df)
        processed_dfs.append(processed_df)
        
    df_train_features = pd.concat(processed_dfs, ignore_index=True)
    df_train_features = df_train_features.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    
    # Kaydet
    output_features = PROCESSED_DIR / "features_v2.parquet"
    df_train_features.to_parquet(output_features, index=False)
    print(f"Eğitim öznitelikleri kaydedildi: {output_features} (Satır: {len(df_train_features)})")
    
    # 2. Holdout İstasyonlarını Oku, Hesapla ve Kaydet
    print("\nHoldout istasyonlarının fiziksel öznitelikleri hesaplanıyor...")
    for st_id in ["station02", "station09"]:
        holdout_path = HOLDOUT_DIR / f"{st_id}.parquet"
        if holdout_path.exists():
            df_holdout = pd.read_parquet(holdout_path)
            df_holdout_features = calculate_station_features(df_holdout)
            # Üzerine yaz (overwrite)
            df_holdout_features.to_parquet(holdout_path, index=False)
            print(f"Holdout istasyonu öznitelikleri güncellendi: {holdout_path} (Satır: {len(df_holdout_features)})")
        else:
            print(f"Uyarı: Holdout {st_id}.parquet bulunamadı.")
            
    print("\nSTAGE-3 başarıyla tamamlandı!")
    print("Sütun Tipleri ve Öznitelik Özeti:")
    print(df_train_features.info())


if __name__ == "__main__":
    main()
