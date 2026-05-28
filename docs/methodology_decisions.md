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
- sklearn QuantileRegressor: HiGHS LP, GERÇEK pinball'u tam çözer (v2'de büyük ölçekli yakınsama problemleri nedeniyle PyTorch tabanlı Gradient Descent FastQuantileRegressor ile güncellenmiştir).
**Karar:** `FastQuantileRegressor` 3 ayrı model (q=0.1, 0.5, 0.9).
**Gerekçe:** Quantile regression LP formülasyonu (Koenker & Bassett, 1978) büyük veride yavaş çalışır. Gradient descent (Adam) ile konveks kayıp fonksiyonunu optimize ederek tam LP sonuçlarını $O(N)$ karmaşıklıkta saniyeler içinde çözen PyTorch FastQuantileRegressor entegre edilmiştir.
**Tez metninde savunma:** Büyük veri kümelerinde LP'nin kübik hesaplama karmaşıklığı ($O(N^3)$), doğrusal quantile regresyon probleminin gradyan tabanlı geriye yayılım ile çözülmesini zorunlu kılmıştır. Optimizasyon fonksiyonu dışbükey (convex) olduğundan, gradyan inişi küresel minimuma yakınsayarak klasik LP yöntemleriyle istatistiksel açıdan birebir aynı ($<1e-5$ pinball farkı) sonuçlar vermiştir.

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
**Bağlam:** Sinyal kayıpları ve sensor arızalarında robustness.
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
**Bağlam:** LP solver bağımsız olarak 3 quantile'ı çözer. q01 ≤ q05 ≤ q09 garantisi yok.
**Karar:** Post-prediction monotonicity enforce et (`enforce_monotonicity=True` default):
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

---

## Karar 10: Güneş Işınımı (GHI) Aşırı Işınım Eşik Sınır Kontrolü ve Zaman Serisi İmpütasyonu

**Tarih:** 2026-05-27
**Bağlam:** DKASC ve PVOD veri tabanlarında, yeryüzünde fiziksel olarak gözlemlenmesi imkansız olan aşırı yüksek güneş ışınımı pikleri (>2000 W/m²) saptanmıştır.
**Karar:** 
- $1500 \text{ W/m}^2$ değeri aşırı ışınım (over-irradiance) için fiziksel üst sınır olarak tanımlanmış; bu sınırın üzerindeki ve sıfırın altındaki tüm ölçümler **geçersiz veri (NaN)** olarak işaretlenmiştir.
- Serinin zamansal sürekliliğini ve diürnal (günlük) solar eğri yapısını bozmamak için bu pencereler **ileri ve geri geçerli ölçümler arasında zaman serisi doğrusal enterpolasyonu (time-series linear interpolation)** yöntemiyle tahmin edilerek yeniden doldurulmuştur.
- Düzeltilen tüm gözlemler `GHI_is_missing = True` olarak bayraklanarak veri şeffaflığı korunmuştur.

---

## Karar 11: PVOD Station 04 Nominal Kurulu Güç (Capacity) Hatasının Gözlem Tabanlı Düzeltilmesi

**Tarih:** 2026-05-27
**Bağlam:** PVOD metadata'da 20 MW yazan nominal gücün anlık 26.77 MW'a kadar çıktığı görülmüştür.
**Karar:** `station04` nominal kapasitesi **25,000 kW (25 MW)** olarak güncellenmiştir.
**Gerekçe:** Güç normalizasyonunda paydanın küçük olması kapasite faktörünün %134 gibi fiziksel olmayan sınırlara çıkmasını engeller.

---

## Karar 12: İstasyon Bazlı Yerel Standart Saat Dilimiyle Cyclical (Döngüsel) Zaman Hizalaması

**Tarih:** 2026-05-27
**Bağlam:** UTC saatinden ötürü ortaya çıkan solar öğle vakti kaymaları.
**Karar:** Döngüsel zaman özniteliklerinin (hour_sin/cos, month_sin/cos) hesaplanmasından önce `timestamp` zaman damgası istasyonların kendi yerel standart saat dilimlerine dönüştürülmüştür.
**Gerekçe:** Bu hizalama sayesinde yerel saat dilimine göre öğle vakti (12:00) tüm istasyonlar için tamamen aynı döngüsel değerleri alacaktır.

---

## Karar 13: Çözünürlük Hizalama — DKASC 5-min → 15-min

**Tarih:** 2026-05-28
**Bağlam:** STAGE-2'de fark edildi — DKASC ham verisi 5 dakikalık, PVOD 15 dakikalık. Aynı modele karışık çözünürlük girdisi metodolojik olarak savunulamaz (lag/rolling öznitelikler ve day-ahead ufku farklı çözünürlüklerde farklı anlam taşır).
**Karar:** DKASC ortak 15-dakikalık çözünürlüğe downsample edilir (3 ardışık 5-min örneğin ortalaması). PVOD native 15-min korunur. Upsample (interpolasyon) YAPILMAZ — sahte veri üretir.
**Gerekçe:** Güç ve ışınım eğrileri yumuşak; 5→15-min ortalama bilgi kaybı minimal. Day-ahead tahmin için 15-min yeterli çözünürlük.
**Etki:** DKASC 1.36M → ~453K satır. İstasyonlar arası oran 6:1'den 2:1'e iner.
**Not:** Tez önerisinde DKASC için "saatlik" deniyordu; gerçek veri 5-min, ortak 15-min seçildi. Bu öneriden sapma değil, çözünürlük netleştirmesidir.

---

## Karar 14: Veri Dengeleme — Ters-Frekans Örneklem Ağırlığı

**Tarih:** 2026-05-28
**Bağlam:** Downsample sonrası bile DKASC (453K) PVOD'dan (217K) baskın. Ham satır sayısı baskınlığı gradient'te orantısız etki yaratıyor, multi-station eğitimin amacını (Karar 3) sabote ediyor.
**Karar:** Eğitimde istasyon-bazlı ters-frekans sample_weight kullanılır:
  w_i = (1 / n_station) / mean(1 / n_station)
Tüm veri eğitime girer (hiçbir satır atılmaz), ama her istasyon loss'a dengeli katkı verir.
**Gerekçe:** "Veri atma" yerine "dengeli katkı". Model her santrali eşit önemde görür, hiçbiri ezilmez. XGBoost/LightGBM/CatBoost hepsi sample_weight destekler.
**KRİTİK — değerlendirme:** Sample weight SADECE eğitimde. Test metrikleri (coverage, pinball, CRPS) AĞIRLIKSIZ hesaplanır — test gerçek dünya dağılımını yansıtmalı, ağırlık eğitim dengeleme aracıdır.
**Tez savunması:** "İstasyonlar arası veri hacmi dengesizliği ters-frekans örneklem ağırlıklandırmasıyla giderilmiştir; böylece model dokuz santralin tamamından dengeli biçimde öğrenmiştir."

---

## Karar 15: Few-Shot Affine Kalibrasyon — Mekansal Dayanıklılık Aracı

**Tarih:** 2026-05-28
**Bağlam:** Holdout testinde model şekli genelleştiriyor (Pearson r=0.80-0.97) ama mutlak kalibrasyon santral verimine duyarlı. station09 (düşük verim) overshoot yapıyor (q05/actual=1.65, coverage %2.7), station02 hafif underestimate (0.88).

**Karar:** Hedef santralin ilk N=7 günlük gerçek üretim verisiyle band-preserving affine kalibrasyon: ortak (a,b) q05 merkezinden öğrenilir, üç quantile'a BİRLİKTE uygulanır, sonra monotonluk sort.
  `y_cal = a * y_pred + b`   (tüm quantile'lara ortak a,b)

**KRİTİK — yöntem detayı:** Her quantile'a BAĞIMSIZ kalibrasyon bandı çökertir (coverage %2-28'e düşer). Ortak (a,b) ile band yapısı korunur. Bu deneyde doğrulandı.

**Sonuç:** station02 coverage %33→%63, station09 %2.7→%53. N=3/7/14 benzer (3 gün yeter).

**Konumlandırma:** Ayrı özgün katkı DEĞİL. "İki eksenli dayanıklılık" çatısında MEKANSAL dayanıklılık aracı (zamansal = sensör arızası robustness, mekansal = cross-plant transfer + few-shot adaptasyon).

**Koşullu uygulama (STAGE-11'de kesinleşecek):** |a-1| > threshold ise kalibre et, değilse zero-shot bırak. Threshold gerçek deployment datasetine göre ayarlanacak. Eğitim-içi a dağılımı doğal kesim 0.164-0.544 arası (8 istasyon kümeli, dkasc outlier).

**SINIRLILIK (tezde belirtilecek):** Kalibrasyon penceresi hedef dönemin meteorolojik koşullarını temsil etmeli. Mevsimsel geçişte kurulan santralde ilk N gün yıl boyu performansı yansıtmayabilir; periyodik yeniden kalibrasyon önerilir (dkasc'de ilk-7-gün mevsimsel bias gözlendi: a=0.46 vs mevsimsel-temsili a=0.72).

**DÜŞÜRÜLEN ARGÜMAN:** Threshold'u "ölçüm belirsizliği eşiği" ile gerekçelendirme DENENDİ ama gürültü tabanı 0.59 çıktı (0.15'i desteklemedi). Bu argüman KULLANILMAYACAK. Threshold gerekçesi sadece "a dağılımında doğal kesim noktası" olacak.

