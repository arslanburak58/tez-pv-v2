# Yöntemsel Kararlar — tez-pv-v2

Bu dosya her önemli yöntemsel kararı numaralı olarak kaydeder. Karar verildiğinde:
gerekçe, alternatifler, dezavantajlar, ileride değiştirme koşulu.

---

## Karar 1: Kapasite Normalizasyonu (ZORUNLU değişiklik v1'den)

**Tarih:** 2026-05-27
**Bağlam:** v1'de mutlak kW değerleriyle eğitildi. Sonuç: model DKASC'nin 150 kW tavanını öğrendi, 17 MW santral'e uygulandığında genelleme başarısız (q50 ≈ 0 kW, oysa actual ~10 MW).
**Karar:** Hedef değişken `y_norm = power_kW / capacity_kW ∈ [0,1]` olacak.
**Gerekçe:** Multi-plant transfer için tek doğru yaklaşım. Model meteorolojik koşulları kapasite faktörüyle ilişkilendirir, mutlak güçle değil.
**Alternatifler:** (a) Per-plant model eğitmek (10 ayrı model, paylaşım yok) — reddedildi. (b) Mutlak kW + scale-adjustment heuristik — reddedildi (aldatmaca).
**Sonuç:** Beklentiler — DKASC ve PVOD aynı modelde eğitilebilir, holdout istasyonda meaningful tahmin.

---

## Karar 2: Meta-Öğrenici — Ridge yerine sklearn QuantileRegressor

**Tarih:** 2026-05-27
**Bağlam:** Tez önerisinde "Ridge meta-learner" yazıyor. v1'de QuantileLinearBounded kullanıldı (custom). Her ikisi de quantile için optimum değil.
- Ridge: nokta tahmin için, MSE minimize eder.
- QuantileLinearBounded: pinball yaklaşımı, q01 collapse.
- sklearn QuantileRegressor: HiGHS LP, gerçek pinball'u tam çözer.
**Karar:** `sklearn.linear_model.QuantileRegressor(solver='highs', alpha=0.0)` 3 ayrı model (q=0.1, 0.5, 0.9).
**Gerekçe:** Quantile regression LP formülasyonu (Koenker & Bassett, 1978):
  minimize Σ ρ_q(y - Xβ)
HiGHS solver bu LP'yi optimum çözer; yaklaşım yok. q01 collapse problemini yapısal olarak engeller. CQR post-hoc kalibrasyonuna gerek kalmaz.
**Tez metninde savunma:** Geliştirme sürecinde Ridge regresyonun nokta tahmin için optimize edildiği, quantile hedefleri için lineer quantile regresyonunun (Koenker & Bassett, 1978) daha uygun olduğu görülmüştür. Stacking mimarisi (Wolpert, 1992) ve metodoloji aynı; yalnızca meta-öğrenici loss fonksiyonu Ridge'in MSE'sinden pinball'a güncellenmiştir.
**Alternatifler:** (a) Ridge + CQR — v1'de denendi, sub-optimum. (b) Quantile Lasso (alpha>0) — eğer overfitting görülürse denenebilir.
**Sonuç:** Beklenti — coverage hedefine direkt ulaşım, q01 collapse yok.

---

## Karar 3: Çoklu İstasyon Eğitimi + Holdout Generalization Testi

**Tarih:** 2026-05-27
**Bağlam:** v1'de sadece DKASC eğitildi, PVOD test olarak kullanılmadı. Tez önerisinde her iki veri seti de var.
**Karar:** 
- Eğitim: DKASC Alice Springs + PVOD v1.0 station00, station01, station03, station04, station05, station06, station07, station08 (8 istasyon)
- Standart test: Her istasyonun son %20 zamanı (kronolojik, walk-forward)
- Generalization test: PVOD station02 + station09 (eğitime HİÇ girmez)
**Gerekçe:** 
- Generalization gerçek bir test ister. Aynı istasyonun son %20'si "near-future" demektir, "yeni santral" demek değil.
- station02 (Mono-Si, tek): farklı panel teknolojisi
- station09 (Poly-Si, normal): aynı teknoloji farklı lokasyon
- İkisi birlikte "panel teknolojisi" ve "lokasyon" eksenlerini test eder.
**Alternatifler:** (a) Tüm PVOD istasyonlarını eğitime kat, holdout yok — reddedildi (gerçek genelleme testi olmaz). (b) Tek holdout (sadece station02) — reddedildi (yetersiz örneklem).
**Sonuç:** Beklenti — eğitim/standart test coverage ≥0.80, holdout coverage 0.65-0.80 (kısmi degradation kabul).

---

## Karar 4: Walk-Forward CV — Gap = 24 saat

**Tarih:** 2026-05-27
**Bağlam:** Zaman serisi CV'de val/test seti train sonrası, ama bitişik. Bitişiklik leakage riski yaratır (örn. son train satırı 23:59:55, ilk val satırı 00:00:00).
**Karar:** `TimeSeriesSplit(n_splits=5, gap=24*4)` — 24 saatlik gap (15-min veri için 96 örnek).
**Gerekçe:** Tahmin ufkumuz day-ahead (24 saat). Train ile val arasında 24 saatlik boşluk, leakage'i fiziksel olarak imkansızlaştırır.
**Sonuç:** Slightly daha az veri ama leakage riski sıfır.

---

## Karar 5: Eksiklik Bayrakları — Sadece Meta-Katmanda

**Tarih:** 2026-05-27
**Bağlam:** Klasik literatür (Jones 1996) missingness indicator'ları nedensellik için eleştirir. Ama tahmin için (Sperrin 2020) destekler.
**Karar:** Bayrakları base model girdisine değil, sadece meta-öğrenici girdisine ekle.
**Gerekçe:** Base modeller saf meteorolojik öğrenir; meta-katmanda flag bilgisi modelin "bu tahminden ne kadar emin olmalıyım" kararına girer.
**Sonuç:** Robustness senaryolarında daha kararlı performans (v1'de gözlemlendi).

---

## Karar 6: Power Birimleri ve Pipeline Boyu Tutarlılık

**Tarih:** 2026-05-27
**Bağlam:** v1'de PVOD power'ın MW olduğu geç fark edildi → app'te ölçek hatası.
**Karar:** Veri ETL aşamasında HER istasyonun power'ı kW'a normalize ediliyor + metadata'da capacity_kW. Sonra y_norm = kW/capacity.
**Doğrulama testi:** Her istasyon için `assert 0 <= y_norm <= 1.5` (1.5 toleransı temporary spikes için).
**Sonuç:** Birim hataları imkansız.

---

## Karar 7: Tahmin Çıktısı Birimi

**Tarih:** 2026-05-27
**Bağlam:** Model normalize ([0,1]) tahmin üretir. Final user mutlak kW istiyor.
**Karar:** İki API:
- `predict_normalized(X)` → [0,1] aralığında DataFrame
- `predict_kW(X, capacity_kW)` → mutlak kW (normalize * capacity)
**Gerekçe:** Demo ve değerlendirme için ikisi de gerekiyor.

---

## Karar 8: Quantile Crossing Kontrolü

**Tarih:** 2026-05-27
**Bağlam:** LP solver bağımsız olarak 3 quantile'ı çözer. Teorik olarak q01 ≤ q05 ≤ q09 garantisi yok.
**Karar:** Post-prediction monotonicity enforce et:
```python
preds_sorted = np.sort(preds.values, axis=1)
preds = pd.DataFrame(preds_sorted, columns=['q_0.1', 'q_0.5', 'q_0.9'])
```
**Gerekçe:** Sorting her satıra ayrı uygulanır, model çıktısının doğal sıralamasını bozmaz ama crossing varsa düzeltir.
**Alternatif:** Constrained QR (statsmodels) — daha yavaş, gerek yok.

---

## Karar 9: Robustness Senaryoları — v1'le Aynı

**Tarih:** 2026-05-27
**Bağlam:** Eksik veri simülasyonu için 3 senaryo.
**Karar:**
- Rastgele: %10, %20, %30, %50 oranında rastgele
- Burst: 1, 6, 24 saatlik ardışık pencereler
- Sensör-özel: G, T_amb, RH ayrı ayrı
**Toplam:** 9 senaryo (4+3+2 sensör için top farkı). Her birinde model degradation ölçülür.
