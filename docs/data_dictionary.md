# Veri Sözlüğü (Data Dictionary) — tez-pv-v2

Bu dosya, STAGE-2 sonucunda oluşturulan birleşik eğitim veri kümesi (`dataset_v2.parquet`) ve holdout istasyon dosyalarının sütun bazlı tanımlarını, birimlerini, değer aralıklarını ve eksik veri durumlarını içerir.

---

## Genel Yapı (Schema)

Tüm veri kümeleri (eğitim ve holdout) aşağıdaki şemaya birebir uymaktadır:

| Sütun | Veri Tipi | Birim | Range (Değer Aralığı) | Eksik Veri (Null) % | Açıklama |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **timestamp** | datetime64[ns, UTC] | UTC | 2010-01-01 / 2022-12-31 | 0% | DatetimeIndex (Zaman damgası, timezone-aware) |
| **station_id** | object (string) | - | "dkasc_alice_springs", "station00".. | 0% | İstasyon tanımlayıcı adı |
| **lat** | float64 | Derece | [-90.0, 90.0] | 0% | İstasyon enlem koordinatı |
| **lon** | float64 | Derece | [-180.0, 180.0] | 0% | İstasyon boylam koordinatı |
| **capacity_kW** | float64 | kW | 265.13 - 35000.0 | 0% | İstasyon nominal DC kurulu gücü |
| **power_kW** | float64 | kW | 0.0 - 35118.1 | 0% | İstasyon anlık güç üretimi (KW) |
| **y_norm** | float64 | Oran | [0.0, 1.5] | 0% | Kapasite-normalize hedef değişken (power_kW / capacity_kW) |
| **GHI** | float64 | W/m² | 0.0 - 3000.0 | 0% | Küresel yatay ışınım (Global Horizontal Irradiance) |
| **T_amb** | float64 | °C | -23.9 - 61.92 | 0% | Ortam sıcaklığı (Ambient Temperature) |
| **RH** | float64 | % | [0.0, 100.0] | 0% | Bağıl Nem (Relative Humidity) |
| **GHI_is_missing** | bool | - | True / False | 0% | GHI verisinin ham tabloda eksik olup olmadığının bayrağı |
| **T_amb_is_missing**| bool | - | True / False | 0% | T_amb verisinin ham tabloda eksik olup olmadığının bayrağı |
| **RH_is_missing** | bool | - | True / False | 0% | RH verisinin ham tabloda eksik olup olmadığının bayrağı |

---

## Veri Kaynağı Detayları ve Doğrulamalar

### 1. DKASC Alice Springs (Avustralya)
- **Power Birimi:** Ham aktif güç verileri `kW` cinsindendir ve doğrudan kullanılmıştır.
- **Eşdeğer Kapasite Tespiti:** DKASC kurulu kapasitesi ham veride açıkça tanımlanmadığından, veri kümesindeki maksimum güç değeri olan **241.03 kW** formüle sokulmuştur: 
  `capacity_kW = 241.03 * 1.1 = 265.13 kW` (~0.90 peak kapasite faktörü varsayımıyla).
- **Zaman Dilimi:** Ham veride yerel saat diliminde (Australia/Darwin, UTC+9.5) olan zaman damgası, timezone-aware UTC'ye dönüştürülmüştür.
- **Temizlik & İmpütasyon:**
  - `T_amb` sütunundaki arıza/boşluk göstergesi olan **-39.99** değerleri ile <-20 ve >60 olan anormal değerler NaN yapılmış ve lineer enterpolasyon ile doldurulmuştur.
  - `RH` ve `GHI` anormal değerleri temizlenerek enterpolasyon yapılmıştır.

### 2. PVOD v1.0 (Hebei, Çin)
- **Power Birimi MW -> kW:** Ham PVOD veri setindeki `power` sütununun **MW** biriminde olduğu doğrulanmış ve `power_kW = power * 1000` formülüyle **kW** cinsine dönüştürülmüştür. 
- **Zaman Dilimi:** Ham date_time verileri doğrudan UTC olarak okunup işaretlenmiştir.
- **Eksik Veri:** PVOD veri seti tamamen temiz olup (sıfır null), missingness bayrakları `False` olarak işaretlenmiştir.

### 3. Hariç Tutulan Değişkenler
- **Rüzgar Hızı (WS):** DKASC veri setindeki 6 yılı aşkın kesintisiz rüzgar hızı veri boşluğu (%100 eksiklik) ve toplamda rüzgar verisinin sadece 2 yıllık sınırlı bir periyodu kapsaması nedeniyle, kullanıcı isteği üzerine **WS ve ilgili tüm eksiklik bayrakları pipeline'dan tamamen çıkarılmıştır.**
