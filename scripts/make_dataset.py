"""
STAGE-2: Veri ETL ve Kapasite Normalizasyonu

Bu script:
1. DKASC + PVOD ham verilerini okur.
2. Rüzgar hızı (WS) verilerini tamamen devre dışı bırakır (kullanıcı isteği doğrultusunda).
3. PVOD power birimini doğrular ve kW'a çevirir (MW -> kW).
4. DKASC Alice Springs verilerini temizler, eksiklikleri enterpolasyon ile tamamlar ve bayraklandırır.
5. DKASC kapasitesini y_test.max() * 1.1 formülüyle dinamik hesaplar.
6. Her satır için y_norm = power_kW / capacity_kW hesaplar.
7. Holdout istasyonlarını ayırır (station02, station09).
8. Birleşik DataFrame'i parquet'e kaydeder.

Çıktı:
- data/processed/dataset_v2.parquet (training data)
- data/processed/holdout/station02.parquet
- data/processed/holdout/station09.parquet
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# Reprodüksiyon için tohumlar
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
HOLDOUT_DIR = OUT_DIR / "holdout"
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_STATIONS = ["station02", "station09"]


def verify_pvod_power_unit(df: pd.DataFrame, capacity_kW: float) -> str:
    """PVOD power sütununun kW mı MW mı olduğunu tespit et.

    Parameters
    ----------
    df : pd.DataFrame
        İstasyon ham veri çerçevesi.
    capacity_kW : float
        İstasyon nominal kapasitesi (kW).

    Returns
    -------
    str
        "kW" veya "MW" birim göstergesi.
    """
    max_power = df["power"].max()
    cf = max_power / capacity_kW
    
    if 0.05 < cf < 1.05:
        return "kW"
    elif 0.00005 < cf < 0.00105:
        return "MW"
    else:
        # toleranslı kontrol: 1.5'e kadar spike'lara izin ver
        if 0.05 < cf < 1.55:
            return "kW"
        elif 0.00005 < cf < 0.00155:
            return "MW"
        raise ValueError(
            f"Power birimi tespit edilemedi. "
            f"max={max_power}, capacity={capacity_kW}, cf={cf}"
        )


def process_dkasc() -> pd.DataFrame:
    """DKASC Alice Springs ham verilerini okur, temizler ve normalize eder.

    Returns
    -------
    pd.DataFrame
        Temizlenmiş ve normalize edilmiş DKASC veri çerçevesi.
    """
    print("DKASC verileri yükleniyor ve temizleniyor...")
    dk_files = sorted((RAW_DIR / "dkasc").glob("dkasc_*.csv"))
    
    if not dk_files:
        raise FileNotFoundError("DKASC raw dosyaları bulunamadı.")
        
    cols_to_use = [
        "timestamp",
        "101_DKA_WeatherStation_Global_Horizontal_Radiation",
        "101_DKA_WeatherStation_Weather_Temperature_Celsius",
        "101_DKA_WeatherStation_Weather_Relative_Humidity",
        "96_DKA_MasterMeter1_Active_Power"
    ]
    
    dfs = []
    for f in tqdm(dk_files, desc="DKASC Yılları"):
        df_year = pd.read_csv(f, usecols=cols_to_use)
        dfs.append(df_year)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Zaman dilimi lokalizasyonu ve UTC dönüşümü
    # DKASC zamanları tz-naive okunduktan sonra Australia/Darwin (UTC+9.5) olarak localize edilir
    # ve ardından UTC'ye dönüştürülür.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Australia/Darwin'de DST (yaz saati) olmadığından ambiguous/nonexistent problemi yoktur.
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize("Australia/Darwin", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    
    # 2. Değişken isimlerini standartlaştır
    df = df.rename(columns={
        "101_DKA_WeatherStation_Global_Horizontal_Radiation": "GHI",
        "101_DKA_WeatherStation_Weather_Temperature_Celsius": "T_amb",
        "101_DKA_WeatherStation_Weather_Relative_Humidity": "RH",
        "96_DKA_MasterMeter1_Active_Power": "power_kW"
    })
    
    # 3. Anormal Değerleri Temizleme (Outlier Filtering)
    # GHI: < -50 veya > 3000 -> NaN
    df.loc[(df["GHI"] < -50) | (df["GHI"] > 3000), "GHI"] = np.nan
    df["GHI"] = df["GHI"].clip(lower=0, upper=1500) # Fiziksel limit [0, 1500] ile clip et (Spike filtreleme)
    
    # T_amb: < -20 veya > 60 -> NaN (Sensör arızası -39.99'lar buraya girer)
    df.loc[(df["T_amb"] < -20) | (df["T_amb"] > 60), "T_amb"] = np.nan
    
    # RH: < 0 veya > 100 -> NaN
    df.loc[(df["RH"] < 0) | (df["RH"] > 100), "RH"] = np.nan
    df["RH"] = df["RH"].clip(lower=0, upper=100)
    
    # power_kW: Negatif değerleri 0'a çek
    df["power_kW"] = df["power_kW"].clip(lower=0)
    
    # 4. Eksiklik Bayraklarını (Missingness Flags) Oluştur
    df["GHI_is_missing"] = df["GHI"].isna()
    df["T_amb_is_missing"] = df["T_amb"].isna()
    df["RH_is_missing"] = df["RH"].isna()
    
    # 5. Eksik Meteorolojik Verileri İmpüte Et (Lineer Enterpolasyon)
    # Forward-fill + backward-fill ile kenar durumları çözülür
    df["GHI"] = df["GHI"].interpolate(method="linear").ffill().bfill()
    df["T_amb"] = df["T_amb"].interpolate(method="linear").ffill().bfill()
    df["RH"] = df["RH"].interpolate(method="linear").ffill().bfill()
    
    # 6. Kapasite Normalizasyonu
    # DKASC eq. capacity estimation: max(power_kW) * 1.1 (kapasite faktörü ~0.9 varsayımı)
    max_power = df["power_kW"].max()
    capacity_kW = float(max_power * 1.1)
    df["capacity_kW"] = capacity_kW
    df["y_norm"] = df["power_kW"] / capacity_kW
    df["y_norm"] = df["y_norm"].clip(lower=0.0, upper=1.5)
    
    # 7. Sabit Sütunları ve İstasyon Bilgilerini Ekle
    df["station_id"] = "dkasc_alice_springs"
    df["lat"] = -23.762
    df["lon"] = 133.875
    
    # Target NaN'ları temizle
    initial_len = len(df)
    df = df.dropna(subset=["power_kW", "y_norm"]).reset_index(drop=True)
    dropped_count = initial_len - len(df)
    
    # Sadece gerekli sütunları seç
    final_cols = [
        "timestamp", "station_id", "lat", "lon", "capacity_kW", "power_kW", "y_norm",
        "GHI", "T_amb", "RH", "GHI_is_missing", "T_amb_is_missing", "RH_is_missing"
    ]
    
    print(f"DKASC tamamlandı. Hesaplanan Kapasite: {capacity_kW:.2f} kW. Toplam Satır: {len(df)} (Dropped target NaNs: {dropped_count})")
    return df[final_cols]


def process_pvod() -> list[pd.DataFrame]:
    """PVOD v1.0 istasyon ham verilerini okur, temizler ve normalize eder.

    Returns
    -------
    list[pd.DataFrame]
        İstasyon bazında temizlenmiş ve normalize edilmiş veri çerçeveleri listesi.
    """
    print("PVOD verileri yükleniyor ve temizleniyor...")
    metadata_path = RAW_DIR / "pvod" / "datasets" / "metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError("PVOD metadata.csv bulunamadı.")
        
    df_meta = pd.read_csv(metadata_path)
    
    station_dfs = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="PVOD İstasyonları"):
        st_id = row["Station_ID"]
        cap_kW = float(row["Capacity"])
        
        # station04 kapasite düzeltmesi (ham veride yıl boyu >20 MW değerler var, gerçek kapasite 25 MW)
        if st_id == "station04":
            cap_kW = 25000.0
            
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        
        st_file = RAW_DIR / "pvod" / "datasets" / f"{st_id}.csv"
        if not st_file.exists():
            print(f"Uyarı: {st_id} dosyası bulunamadı, atlanıyor.")
            continue
            
        # Gerekli sütunları oku (rüzgar hızı WS tamamen hariç tutuluyor)
        cols_to_use = [
            "date_time",
            "lmd_totalirrad",
            "lmd_temperature",
            "nwp_humidity",
            "power"
        ]
        
        df = pd.read_csv(st_file, usecols=cols_to_use)
        
        # 1. Zaman Damgası UTC İşaretleme
        # PVOD raw verilerindeki date_time UTC zaman dilimindedir
        df["timestamp"] = pd.to_datetime(df["date_time"]).dt.tz_localize("UTC")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 2. Değişken İsimlerini Standartlaştır
        df = df.rename(columns={
            "lmd_totalirrad": "GHI",
            "lmd_temperature": "T_amb",
            "nwp_humidity": "RH"
        })
        
        # 3. Güç Birimi Doğrulama ve kW'a Dönüştürme (MW -> kW)
        unit = verify_pvod_power_unit(df, cap_kW)
        if unit == "MW":
            df["power_kW"] = df["power"] * 1000.0
        else:
            df["power_kW"] = df["power"]
            
        df["power_kW"] = df["power_kW"].clip(lower=0)
        
        # 4. Kapasite Normalizasyonu
        df["capacity_kW"] = cap_kW
        df["y_norm"] = df["power_kW"] / cap_kW
        df["y_norm"] = df["y_norm"].clip(lower=0.0, upper=1.5)
        
        # 5. Meteorolojik Sınır Kontrolleri ve Temizlik (PVOD verisi genellikle null barındırmaz)
        df["GHI"] = df["GHI"].clip(lower=0.0, upper=1500.0) # Fiziksel limit [0, 1500] ile clip et (Spike filtreleme)
        df["RH"] = df["RH"].clip(lower=0.0, upper=100.0)
        
        # Eksiklik Bayraklarını Oluştur (PVOD temiz olduğundan hepsi False olacaktır)
        df["GHI_is_missing"] = False
        df["T_amb_is_missing"] = False
        df["RH_is_missing"] = False
        
        # Sabit Sütunları Ekle
        df["station_id"] = st_id
        df["lat"] = lat
        df["lon"] = lon
        
        # Target NaN'ları temizle
        df = df.dropna(subset=["power_kW", "y_norm"]).reset_index(drop=True)
        
        final_cols = [
            "timestamp", "station_id", "lat", "lon", "capacity_kW", "power_kW", "y_norm",
            "GHI", "T_amb", "RH", "GHI_is_missing", "T_amb_is_missing", "RH_is_missing"
        ]
        
        station_dfs.append(df[final_cols])
        
    return station_dfs


def main() -> None:
    """ETL boru hattını çalıştırır, holdout'ları ayırır ve birleşik dataset'i kaydeder."""
    # 1. DKASC verilerini işle
    df_dkasc = process_dkasc()
    
    # 2. PVOD verilerini işle
    pvod_dfs = process_pvod()
    
    # 3. Holdout İstasyonlarını Ayır ve Kaydet
    train_pvod_dfs = []
    
    for df in pvod_dfs:
        st_id = df["station_id"].iloc[0]
        if st_id in HOLDOUT_STATIONS:
            out_file = HOLDOUT_DIR / f"{st_id}.parquet"
            df.to_parquet(out_file, index=False)
            print(f"Holdout istasyonu kaydedildi: {out_file} (Satır: {len(df)})")
        else:
            train_pvod_dfs.append(df)
            
    # 4. Eğitim İstasyonlarını (8 PVOD + 1 DKASC) Birleştir
    print("Eğitim veri kümesi birleştiriliyor...")
    all_train_dfs = [df_dkasc] + train_pvod_dfs
    df_train = pd.concat(all_train_dfs, ignore_index=True)
    
    # Sıralama: timestamp bazında ve ardından station_id bazında sıralı tutalım
    df_train = df_train.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    
    # 5. Kaydet
    output_parquet = OUT_DIR / "dataset_v2.parquet"
    df_train.to_parquet(output_parquet, index=False)
    print(f"Eğitim veri kümesi başarıyla kaydedildi: {output_parquet}")
    print(f"Birleşik Eğitim Satır Sayısı: {len(df_train)}")
    
    # Sütun tiplerini ve null durumlarını kontrol et
    print("\nVeri Tipi ve Eksik Veri Özeti:")
    print(df_train.info())
    print("\nNaN Sayıları:")
    print(df_train.isna().sum())


if __name__ == "__main__":
    main()
