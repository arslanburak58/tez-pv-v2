"""STAGE-4: Veri setindeki değişkenleri 15 günlük ardışık zaman ve 24 saatlik diürnal bindirme (overlay) olarak görselleştiren betik."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Seaborn/Matplotlib görsel estetik ayarları
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Inter"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.5,
    "axes.labelcolor": "#212529",
    "xtick.color": "#495057",
    "ytick.color": "#495057",
    "text.color": "#212529"
})

ROOT = Path("/Users/burakarslan/Desktop/tez-pv-ant")
FEATURES_FILE = ROOT / "data" / "processed" / "features_v2.parquet"
OUTPUT_DIR = Path("/Users/burakarslan/.gemini/antigravity/brain/331603e0-f634-4be3-b3f1-86595f3b262d")


def main() -> None:
    # 1. Veriyi yükle
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} bulunamadı.")
        
    df = pd.read_parquet(FEATURES_FILE)
    
    # 2. DKASC istasyonunun 2020 yılından 15 günlük verisini seç (Yaz mevsimi - yüksek solar volatilite)
    st_df = df[df["station_id"] == "dkasc_alice_springs"].copy()
    
    # Yerel saat damgasına göre sırala ve indeksle
    st_df["local_time"] = st_df["timestamp"].dt.tz_convert("Australia/Darwin")
    st_df = st_df.sort_values("local_time")
    
    # 15 günlük penceremizi seçelim (2020-01-01 ile 2020-01-15 arası)
    start_date = pd.Timestamp("2020-01-01 00:00:00", tz="Australia/Darwin")
    end_date = pd.Timestamp("2020-01-15 23:59:59", tz="Australia/Darwin")
    
    plot_df = st_df[(st_df["local_time"] >= start_date) & (st_df["local_time"] <= end_date)].copy()
    
    # Grafiklerin çizileceği değişkenler ve özellikleri
    variables_to_plot = {
        "y_norm": {
            "title": "Normalized Capacity Factor (Kapasite Faktörü)",
            "ylabel": "Kapasite Faktörü (y_norm)",
            "color": "#1a73e8",  # Vibrant Blue
            "ylim": (-0.05, 1.15)
        },
        "GHI": {
            "title": "Global Horizontal Irradiance (Güneş Işınımı)",
            "ylabel": "Işınım (GHI - W/m²)",
            "color": "#f9ab00",  # Sunny Amber
            "ylim": (-50, 1400)
        },
        "T_amb": {
            "title": "Ambient Temperature (Ortam Sıcaklığı)",
            "ylabel": "Sıcaklık (T_amb - °C)",
            "color": "#ea4335",  # Coral Red
            "ylim": (15, 50)
        },
        "T_cell": {
            "title": "Calculated Cell Temperature (Panel Sıcaklığı)",
            "ylabel": "Sıcaklık (T_cell - °C)",
            "color": "#d93025",  # Dark Crimson
            "ylim": (15, 90)
        },
        "RH": {
            "title": "Relative Humidity (Bağıl Nem)",
            "ylabel": "Bağıl Nem (RH - %)",
            "color": "#12b5cb",  # Teal Cyan
            "ylim": (-5, 105)
        },
        "cos_zenith": {
            "title": "Cosine Zenith Angle (Zenit Açısı Kosinüsü)",
            "ylabel": "cos_zenith",
            "color": "#8ab4f8",  # Light Blue
            "ylim": (-1.05, 1.05)
        },
        "k_t": {
            "title": "Clear-Sky Index (Açık Gökyüzü İndeksi)",
            "ylabel": "k_t",
            "color": "#34a853",  # Leaf Green
            "ylim": (-0.05, 1.35)
        },
        "hour_angle": {
            "title": "Solar Hour Angle (Saat Açısı)",
            "ylabel": "Saat Açısı (Derece)",
            "color": "#ab47bc",  # Lavender Purple
            "ylim": (-190, 190)
        }
    }
    
    # 24 saatlik x-ekseni için yerel saat saatini ve gün bazlı gruplamayı yap
    plot_df["hour_local"] = plot_df["local_time"].dt.hour + plot_df["local_time"].dt.minute / 60.0
    plot_df["date_local"] = plot_df["local_time"].dt.date
    
    # Grafik klasörünün varlığından emin ol
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = pd.io.common.LoggingContext if hasattr(pd.io.common, "LoggingContext") else None
    
    for var, cfg in variables_to_plot.items():
        print(f"{var} değişkeni görselleştiriliyor...")
        
        # 16:9 oranında premium dual-layout (Sol: Zaman Serisi, Sağ: 24 Saatlik Bindirme)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.3, 1]})
        fig.suptitle(f"DKASC Alice Springs - {cfg['title']} (15 Günlük Profil)", fontsize=16, fontweight="bold", color="#202124")
        
        # --- SOL GRAFİK: 15 Günlük Kesintisiz Zaman Serisi ---
        ax0 = axes[0]
        ax0.plot(plot_df["local_time"], plot_df[var], color=cfg["color"], linewidth=1.5, alpha=0.85)
        ax0.set_title("15 Günlük Kesintisiz Zaman Serisi", fontsize=12, fontweight="semibold", color="#5f6368")
        ax0.set_xlabel("Tarih", fontsize=11)
        ax0.set_ylabel(cfg["ylabel"], fontsize=11)
        ax0.set_ylim(cfg["ylim"])
        
        # Tarih x-eksenini güzelleştir
        plt.setp(ax0.get_xticklabels(), rotation=30, ha="right")
        
        # --- SAĞ GRAFİK: 24 Saatlik Overlay Profil ---
        ax1 = axes[1]
        
        # Her bir günü ince ince saydam çizgilerle çiz
        unique_dates = plot_df["date_local"].unique()
        for d in unique_dates:
            day_data = plot_df[plot_df["date_local"] == d]
            ax1.plot(day_data["hour_local"], day_data[var], color=cfg["color"], linewidth=1.0, alpha=0.15)
            
        # 15 günün ortalama eğrisini kalın çizgi olarak çiz
        # Saatlere göre gruplayarak ortalama alalım (5 dakikalık hassasiyetle)
        # Yuvarlatılmış saatlik trend
        mean_profile = plot_df.groupby(plot_df["local_time"].dt.time)[var].mean()
        mean_hours = [t.hour + t.minute/60.0 for t in mean_profile.index]
        
        # Saat sırasına göre sırala
        sort_idx = np.argsort(mean_hours)
        mean_hours = np.array(mean_hours)[sort_idx]
        mean_values = np.array(mean_profile.values)[sort_idx]
        
        ax1.plot(mean_hours, mean_values, color=cfg["color"], linewidth=3.0, label="15 Günlük Ortalama")
        
        ax1.set_title("24 Saatlik Overlay Günlük Profil", fontsize=12, fontweight="semibold", color="#5f6368")
        ax1.set_xlabel("Günün Saati (Yerel Saat dilimi)", fontsize=11)
        ax1.set_ylabel(cfg["ylabel"], fontsize=11)
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 4))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
        ax1.set_ylim(cfg["ylim"])
        ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#eeeeee")
        
        plt.tight_layout()
        
        # Grafiği kaydet
        out_path = OUTPUT_DIR / f"plot_{var.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi: {out_path}")
        
    print("\nTüm değişkenlerin grafikleri başarıyla üretildi!")


if __name__ == "__main__":
    main()
