# Stage Tamamlanma Logu

Her stage tamamlandığında: tarih, ne yapıldı, çıktılar, doğrulama.

---

## 2026-05-27 STAGE-0 — Proje Kurulumu ✓

**Dosyalar:**
- README.md, CLAUDE.md, Makefile, requirements.txt, .gitignore
- docs/methodology_decisions.md (Karar 1-9)
- docs/lessons_learned.md (Ders 1-8)
- docs/tez_workflow.md (STAGE-0 to STAGE-13)
- docs/gunce.md, docs/stage_log.md
- docs/skills/skill_*.md (5 dosya)
- features/, models/, scripts/, tests/, app/ (boş __init__.py'lar)

**GitHub:** arslanburak58/tez-pv-v2 (public)

**Doğrulama:** `make help` çalışıyor, ilk commit push edildi.

---

## 2026-05-27 STAGE-2 — Veri Toplama ve Kapasite Normalizasyonu ✓

**Dosyalar:**
- `data/processed/dataset_v2.parquet` (1,577,952 satır birleşik eğitim verisi)
- `data/processed/holdout/station02.parquet` (30,432 satır holdout verisi)
- `data/processed/holdout/station09.parquet` (24,288 satır holdout verisi)
- `scripts/make_dataset.py` (ETL pipeline scripti)
- `tests/test_dataset.py` (PyTest doğrulamaları)
- `docs/data_dictionary.md` (Veri sözlüğü güncellendi)

**Yapılan İşlemler:**
1. **PVOD Güç Dönüşümü:** Ham verinin **MW** biriminde olduğu doğrulanarak `kW` birimine çevrildi (`* 1000`).
2. **DKASC Kapasite Tahmini:** `max(power_kW) * 1.1` formülüyle DKASC nominal kapasitesi **265.13 kW** olarak hesaplandı.
3. **y_norm Normalizasyonu:** Hedef değişken `y_norm = power_kW / capacity_kW ∈ [0.0, 1.5]` olarak normalize edildi.
4. **Zaman Dilimi Standartlaştırma:** Tüm zaman serileri `UTC` zaman dilimine ve tz-aware DatetimeIndex yapısına dönüştürüldü.
5. **Meteorolojik Temizlik:** Geçersiz/hatalı (T_amb -39.99 gibi) veriler NaN yapılarak lineer enterpolasyon ile giderildi. Eksiklik bayrakları (`_is_missing`) oluşturuldu. Rüzgar hızı (WS) verileri kullanıcı isteğiyle tamamen çıkarıldı.
6. **Holdout Ayrıştırma:** Generalization testi için `station02` ve `station09` veri kümesinden tamamen ayrılarak izole edildi.

## 2026-05-27 STAGE-3 — Fiziksel Öznitelik Mühendisliği ✓

**Dosyalar:**
- `data/processed/features_v2.parquet` (1,577,952 satır öznitelikli eğitim verisi)
- `data/processed/holdout/station02.parquet` (Özniteliklerle güncellendi)
- `data/processed/holdout/station09.parquet` (Özniteliklerle güncellendi)
- `scripts/build_features.py` (pvlib tabanlı öznitelik üretici)
- `tests/test_features.py` (Yeni fiziksel doğrulama testleri)
- `docs/methodology_decisions.md` (Karar 12 - Yerel Saat Hizalaması eklendi)

**Yapılan İşlemler:**
1. **Solar Konum & Açı:** `apparent_zenith`, `zenith`, `apparent_elevation`, `elevation`, `azimuth` ve `cos_zenith` pvlib ile hesaplandı.
2. **Saat Açısı (H):** Kesirli UTC saati, Boylam ve Zaman Denklemi (EoT) kullanılarak Yerel Güneş Saati (LST) üzerinden Saat Açısı $H \in [-180, 180]$ derece olarak tam fiziksel formülle üretildi.
3. **Atmosferik Kütle:** Bağıl airmass pvlib ile hesaplandı ve gece sıfırlanıp gündüz clip edilerek kararlı hale getirildi.
4. **Berraklık İndeksi (k_t):** Dünya dışı yatay ışınım hesaplandı. Gün doğumu/batımı eşiği (zenith < 85°) dışındaki gece saatlerinde $k_t = 0.0$ yapıldı, gündüz ise clip edilerek $[0.0, 2.0]$ aralığında sınırlandırıldı.
5. **Panel Sıcaklığı (T_cell):** NOCT = 45°C için Ross Formülü ($T_{cell} = T_{amb} + 0.03125 \times GHI$) ile hesaplandı.
6. **Yerel Saat Hizalamalı Döngüsel Zaman:** İstasyonların yerel standard saatlerine (Darwin ve Shanghai) dönüştürülmüş saat ve ay değerleri üzerinden döngüsel sin/cos öznitelikleri üretildi, böylece modeller için "solar öğle vakti" ortak örüntüde eşitlendi.

**Doğrulama:** `make test` komutuyla 10 doğrulama testinin tamamı (cos_zenith sınırları, $k_t$ fiziksel aralıkları, Ross sıcaklık doyum testi, döngüsel trigonometrik özdeşlikler) başarıyla geçti.

---


