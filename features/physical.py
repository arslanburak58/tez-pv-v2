"""
pvlib tabanlı fiziksel öznitelikler.

Her satır için (lat, lon, timestamp) bilgisiyle astronomik ve termal
büyüklükler hesaplanır.

STAGE-3'te tam implementasyon yapılacak.
"""
from typing import Any
import numpy as np
import pandas as pd


def compute_cos_zenith(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
) -> pd.Series:
    """Güneş zenit açısının kosinüsünü hesaplar.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Tz-aware timestamps.
    latitude, longitude : float
        Coğrafi konum (decimal degrees).
    altitude : float, optional
        Yükseklik (metre).

    Returns
    -------
    pd.Series
        cos(zenith), gece negatif, gündüz pozitif.
    """
    raise NotImplementedError("STAGE-3'te implement edilecek")


def compute_clearness_index(
    ghi: pd.Series,
    extraterrestrial: pd.Series,
) -> pd.Series:
    """Açıklık indeksi k_t = GHI / G_0."""
    raise NotImplementedError("STAGE-3'te implement edilecek")


def compute_cell_temperature(
    t_amb: pd.Series,
    ghi: pd.Series,
    noct: float = 45.0,
) -> pd.Series:
    """Ross formülü ile hücre sıcaklığı."""
    raise NotImplementedError("STAGE-3'te implement edilecek")


def build_physical_features(
    df: pd.DataFrame,
    location: dict[str, Any],
) -> pd.DataFrame:
    """Tüm fiziksel öznitelikleri üretir.

    Parameters
    ----------
    df : pd.DataFrame
        En az 'T_amb' ve 'G' (GHI) sütunlarını içeren tz-aware DataFrame.
    location : dict
        {'latitude': float, 'longitude': float, 'altitude': float, 'tz': str}

    Returns
    -------
    pd.DataFrame
        Eklenmiş sütunlar: cos_zenith, hour_angle, air_mass, k_t, T_cell,
        hour_sin, hour_cos, month_sin, month_cos
    """
    raise NotImplementedError("STAGE-3'te implement edilecek")
