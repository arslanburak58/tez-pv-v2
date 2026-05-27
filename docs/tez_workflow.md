# Tez Workflow — Stage Yol Haritası

Her stage için: amaç, girdi, çıktı, başarı kriterleri.

---

## STAGE-0: Proje Kurulumu

**Amaç:** Klasör yapısı, git repo, dokümantasyon iskeleti, Python ortamı.

**Çıktı:**
- ~/Desktop/tez-pv-v2/ klasör yapısı (veya ~/Desktop/tez-pv-ant/ çalışma alanı)
- GitHub'da arslanburak58/tez-pv-v2 repo (public veya private)
- Tüm docs/ dosyaları
- .venv aktif, requirements.txt kurulu

**Başarı:** `make help` çıktısı çalışır, ilk commit push edilmiş.

---

## STAGE-1: Literatür Sentezi

**Durum:** v1'den taşındı, ek çalışma gerektirmiyor.

**Girdi:** `docs/literatur_ozeti.md` (v1'den kopyalanır, yöntem değişikliklerine göre güncellenir)

**Başarı:** En az 28 makale APA formatında, her birine iş paketi atanmış, Atiea (2025) karşılaştırması her makalede yapılmış.

---

## STAGE-2: Veri Toplama ve Normalizasyon

**Amaç:** DKASC + PVOD ham verisini çekip kapasite-normalize edilmiş tek bir dataset oluşturmak.

**Girdi:**
- data/raw/dkasc/ (Alice Springs 2010–2022, hourly)
- data/raw/pvod/ (10 istasyon CSV, metadata.csv)

**İşlemler:**
1. PVOD power birimi tespiti (kW mı MW mı?) ve standardize (kW)
2. Her istasyon için capacity_kW lookup (metadata.csv'den)
3. y_norm = power_kW / capacity_kW hesabı
4. DKASC eq. capacity estimation: y_test.max() * 1.1 (kapasite faktörü 0.9 varsayım)
5. Birleşik DataFrame: timestamp, station_id, lat, lon, capacity_kW, power_kW, y_norm, [meteo cols], [is_missing flags]
6. Holdout istasyonları ayır: station02 ve station09
7. Train+val: DKASC + 8 PVOD istasyonu

**Çıktı:** `data/processed/dataset_v2.parquet` (single combined file)

**Tamamlandı kriteri:**
- [ ] Tüm istasyonlar için y_norm ∈ [0, 1.5] (assertion testi)
- [ ] `data/processed/data_dictionary.md` güncel
- [ ] Holdout istasyonları ayrı dosyalarda
- [ ] Birim doğrulaması: rastgele 100 satır manuel kontrol

---

## STAGE-3: Fiziksel Öznitelik Mühendisliği

**Amaç:** pvlib ile her istasyon için astronomik ve termal öznitelikler üretmek.

**Girdi:** dataset_v2.parquet (timestamp + lat + lon + T_amb + G)

**İşlemler:**
1. Her satır için (lat, lon) bilgisiyle pvlib solar position
2. cos_zenith, hour_angle, air_mass
3. k_t = G / G_extraterrestrial (clear-sky index)
4. T_cell = T_amb + k * G (Ross formula, NOCT=45)
5. Döngüsel zaman: hour_sin, hour_cos, month_sin, month_cos

**Çıktı:** `data/processed/features_v2.parquet`

**Tamamlandı kriteri:**
- [ ] Gece saatlerinde cos_zenith < 0
- [ ] Gündüz cos_zenith > 0.087 (zenit < 85°)
- [ ] k_t ortalama 0.5-0.7 arası (Hebei + Alice Springs ortalaması)
- [ ] T_cell - T_amb ortalama ~15-25°C (saturasyonda)

---

## STAGE-4: Walk-Forward Split + Holdout

**Amaç:** Train/val/test bölünmesi, leakage-safe.

**İşlemler:**
1. Holdout istasyonları (station02, station09) ayrı kaydet
2. Train+val+test = 8 PVOD + DKASC kombine
3. TimeSeriesSplit(n_splits=5, gap=24*4) walk-forward CV foldları
4. Final test seti: Her istasyonun son %20 zaman dilimi

**Çıktı:**
- `data/processed/splits.joblib` (fold indices)
- `data/processed/holdout/station02.parquet`
- `data/processed/holdout/station09.parquet`

**Tamamlandı kriteri:**
- [ ] Train ve val arası 24 saatlik gap doğrulandı
- [ ] Hiçbir fold'da timestamp overlap yok
- [ ] Holdout istasyonları kesinlikle dışarıda

---

## STAGE-5: Base Learner Eğitimi

**Amaç:** 9 base model (3 algoritma × 3 quantile).

**İşlemler:**
1. XGBoost (custom pinball objective), LightGBM (objective='quantile'), CatBoost (loss_function='Quantile:alpha=q')
2. Walk-forward folds üzerinde OOF predictions üret
3. Her model joblib serialize

**Çıktı:**
- `data/processed/base_models_v2.joblib` (9 model)
- `data/processed/x_meta_v2.joblib` (OOF tahminler, train+val boyunda)

**Tamamlandı kriteri:**
- [ ] 9/9 model eğitildi
- [ ] OOF pinball her algoritma için raporlandı
- [ ] x_meta sütunları: lgbm_q01/05/09, catboost_q01/05/09, xgboost_q01/05/09

---

## STAGE-6: Meta-Öğrenici (sklearn QR)

**Amaç:** sklearn QuantileRegressor (HiGHS LP) meta-öğrenici.

**İşlemler:**
1. Input: x_meta (9 sütun) + flags (3 sütun) = 12 öznitelik
2. 3 QR modeli (q=0.1, 0.5, 0.9), solver='highs', alpha=0.0
3. Quantile crossing check ve post-sorting

**Çıktı:** `data/processed/meta_models_v2.joblib`

**Tamamlandı kriteri:**
- [ ] q01 collapse yok (öğle saatinde q01/q05 ≥ 0.5)
- [ ] Train coverage 0.80 ± 0.02

---

## STAGE-7: Optuna Optimizasyonu

**Amaç:** Base learner hiperparametrelerini Bayesyen olarak optimize et.

**İşlemler:**
- TPE sampler, MedianPruner
- 100 trial / algoritma
- Objective: mean pinball loss on val set

**Çıktı:** `data/processed/optuna_studies/` + güncellenmiş base_models

**Tamamlandı kriteri:**
- [ ] Best trial pinball %5+ iyileşme (baseline'a göre)
- [ ] Convergence plot kaydedildi

---

## STAGE-8: Robustness Testleri

**Amaç:** 9 sensör arıza senaryosunda model degradation ölçümü.

**Senaryolar:**
1. Random: G %10, %20, %30, %50
2. Burst: G 1h, 6h, 24h
3. Sensor-specific: T_amb %30, RH %30

**Çıktı:** `results/robustness_v2.parquet` (her senaryo için coverage, pinball, MAE)

**Tamamlandı kriteri:**
- [ ] Baseline (clean) coverage ≥ 0.80
- [ ] En kötü senaryoda (G %50) coverage ≥ 0.65
- [ ] Flag ablation: missingness flags olmadan vs ile, DM testi anlamlı

---

## STAGE-9: Baseline Modelleri

**Amaç:** k-NN, SVM, LSTM, TFT karşılaştırma baseline'ları.

**İşlemler:**
- Hepsi aynı feature matrix
- Aynı walk-forward split
- M4'te eğitim, gerekirse Colab T4

**Çıktı:** `data/processed/baselines/` (her model joblib)

**Tamamlandı kriteri:**
- [ ] 4 baseline eğitildi
- [ ] Test setinde MAE, RMSE, Pinball karşılaştırma tablosu

---

## STAGE-10: Final Değerlendirme + Holdout Test

**Amaç:** Tüm metrikleri toplu rapor.

**Metrikler:** MAE, RMSE, Pinball, CRPS, Coverage, Band Width, q01/q05 ratio

**Senaryolar:**
- Standart test (her istasyonun son %20'si)
- Holdout test (station02 + station09)
- Robustness ortalama

**Çıktı:** `results/final_metrics_v2.parquet` + LaTeX tablo

**Tamamlandı kriteri:**
- [ ] Holdout coverage ≥ 0.65 (en az)
- [ ] Stacking baseline'lardan %10+ iyi (Pinball)
- [ ] DM testi anlamlı (stacking vs en iyi baseline)

---

## STAGE-11: Streamlit Demo

**Amaç:** İnteraktif demo.

**Özellikler:**
1. İstasyon seçici (10 PVOD + DKASC)
2. Tarih seçici
3. Sensör arıza slider'ları
4. q10/q50/q90 bant görseli
5. Metrikler

**Çıktı:** `app/app.py`

**Tamamlandı kriteri:**
- [ ] Yerel makinede çalışıyor
- [ ] Tüm istasyonlar için tahmin üretir
- [ ] Demo videosu kaydedildi (savunma için)

---

## STAGE-12: Tez Yazımı

**Bölümler:** Özet, Giriş, Literatür, Yöntem, Bulgular, Tartışma, Sonuç

**Çıktı:** `thesis/main.tex` veya enstitü formatında .docx

---

## STAGE-13: SCI/SCI-E Makale Taslağı

**Hedef dergi:** Solar Energy, Energy Conversion and Management, Applied Energy

**Çıktı:** `paper/main.tex`
