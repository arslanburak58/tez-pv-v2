# Fotovoltaik Sistemlerde Sensör Kayıplarına Dayanıklı Olasılıksal Güç Tahmini

**Yüksek Lisans Tez Uygulaması — v2**

Sivas Cumhuriyet Üniversitesi · YZ ve Veri Bilimi Anabilim Dalı
Öğrenci: Burak Arslan · Danışman: Doç. Dr. Kenan Altun

## Özet

Bu proje, fotovoltaik (PV) santraller için olasılıksal güç tahmini geliştirir.
Çıktı: %10–%90 güven bandı (q01, q50, q09). Sensör arızalarına karşı dayanıklılık
test edilir.

### Mimari

```
[Fiziksel Öznitelikler (pvlib)] ──┐
[Meteorolojik Ölçümler]          ├──→ Base Models (XGBoost/LightGBM/CatBoost × 3 quantile)
[Eksiklik Bayrakları]            ──┘                  │
                                                      ▼
                                          OOF Predictions (9 sütun)
                                                      │
                                          + Missingness Flags
                                                      ▼
                                  sklearn QuantileRegressor (HiGHS LP)
                                                      │
                                                      ▼
                                          q01 / q05 / q09 (normalize)
                                                      │
                                          × capacity_kW
                                                      ▼
                                            Final kW Tahminleri
```

### Veri

- **DKASC Alice Springs** (Avustralya) — 2010–2022, hourly
- **PVOD v1.0** (Çin Hebei) — 10 istasyon, 2018–2019, 15min
  - 8 istasyon: eğitim
  - 2 istasyon: generalization test (eğitime hiç girmez)

### v1'den Farkları

| Konu | v1 | v2 |
|------|----|----|
| Hedef değişken | power_kW (mutlak) | power_kW / capacity_kW (normalize, [0,1]) |
| Eğitim verisi | Sadece DKASC | DKASC + PVOD (8 istasyon) |
| Meta-öğrenici | QuantileLinearBounded (custom) | sklearn QuantileRegressor (HiGHS LP) |
| Coverage kalibrasyonu | Post-hoc CQR k=2.0 | LP doğru çözüm, CQR gerekmez |
| Generalization testi | Yok | 2 holdout PVOD istasyonu |

## Kurulum

```bash
# Python ortamı
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Veri girdileri (v1'den senkronize)
make sync-data

# Tüm pipeline
make all
```

## Stage Yapısı

Detaylı: `docs/tez_workflow.md`

- STAGE-0: Proje kurulumu ✓
- STAGE-1: Literatür ✓ (v1'den taşındı)
- STAGE-2: Veri toplama + kapasite normalizasyonu
- STAGE-3: Fiziksel öznitelik mühendisliği (pvlib)
- STAGE-4: Walk-forward split + holdout istasyonları
- STAGE-5: Base learner eğitimi (9 model)
- STAGE-6: sklearn QuantileRegressor meta-öğrenici
- STAGE-7: Optuna optimizasyonu
- STAGE-8: Robustness testleri (sensör arıza simülasyonu)
- STAGE-9: Baseline modelleri (k-NN, SVM, LSTM, TFT)
- STAGE-10: Değerlendirme + holdout testi
- STAGE-11: Streamlit demo
- STAGE-12: Tez yazımı
- STAGE-13: SCI/SCI-E makale taslağı

## Lisans

MIT
