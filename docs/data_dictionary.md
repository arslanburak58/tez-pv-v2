# Veri Sözlüğü

Her veri setinin sütun-by-sütun tanımı, birim, range, eksik veri durumu.

**STAGE-2 sonunda doldurulacak.**

---

## DKASC Alice Springs

| Sütun | Birim | Range | NULL% | Açıklama |
|-------|-------|-------|-------|----------|
| timestamp | UTC | 2010-01-01 / 2022-12-31 | 0% | DatetimeIndex |
| GHI | W/m² | 0–1400 | TBD | Küresel yatay ışınım |
| ... | | | | |

## PVOD v1.0

| Sütun | Birim | Range | NULL% | Açıklama |
|-------|-------|-------|-------|----------|
| date_time | local | 2018-08-15 / 2019-... | 0% | Hebei UTC+8 |
| power | **kW (DOĞRULANACAK!)** | 0–35000 | 0% | İstasyon güç çıkışı |
| ... | | | | |

**KRİTİK:** STAGE-2'de yapılacak ilk iş: PVOD power birimini doğrulamak.
Hipotez: kW. Test: peak_power vs capacity_kW oranı %0-95 arası olmalı.
Eğer oran %0.001 civarındaysa → MW birimde, ×1000 düzeltmesi.
