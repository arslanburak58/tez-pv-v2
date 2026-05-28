## 2026-05-28 — Few-Shot Kalibrasyon Deneyi Tamamlandı (STAGE-11'e ertelendi)

**>>> DEVAM KOMUTU: Yeni konuşmayı "STAGE-7 Optuna'ya başlıyoruz" ile başlat <<<**

Aktif adım    : STAGE-6 tamam, kalibrasyon deneyi tamam, STAGE-7 bekleniyor
Aktif model   : Sonnet 4.6
Son güncelleme: 2026-05-28
Tıkanıklık    : Yok

Kaynaks (Projects için): https://raw.githubusercontent.com/arslanburak58/tez-pv-v2/main/docs/gunce.md

---

## Okuma listesi (Claude her konuşmada fetch eder)
- https://raw.githubusercontent.com/arslanburak58/tez-pv-v2/main/docs/stage_log.md
- https://raw.githubusercontent.com/arslanburak58/tez-pv-v2/main/docs/methodology_decisions.md
- https://raw.githubusercontent.com/arslanburak58/tez-pv-v2/main/docs/lessons_learned.md
- https://raw.githubusercontent.com/arslanburak58/tez-pv-v2/main/docs/tez_workflow.md

---

## Tamamlanan (bu oturumda)
- **FastQuantileRegressor (PyTorch GD):** sklearn QR (HiGHS LP) yerine, Karar 2-revize.
  - 4 kontrol başarıyla geçti: collapse yok (q01/q05=0.837), seed sabit ($0.00\text{e}+00$ fark), monotonluk post-sort, CPU.
- **STAGE-2 düzeltme:** DKASC 5-min→15-min downsample (Karar 10), sample weight (Karar 11).
  - Dengeli: istasyon başına ağırlıklı katkı eşit (74,424.78 her biri).
  - Standart test coverage %78.03, q01/q05 öğle 0.727.
- **Holdout testi (station02, station09):** zero-shot transfer.
  - station02 (Mono-Si): Pearson r=0.97, coverage %33 (bant dar), q05/actual=0.88.
  - station09 (Poly-Si): Pearson r=0.80, coverage %2.7, OVERSHOOT q05/actual=1.65.
  - Bulgu: şekil mükemmel genelleşiyor, mutlak kalibrasyon santral verimine duyarlı.
- **Few-shot kalibrasyon deneyi (experiments/fewshot-calibration branch):**
  - Band-preserving affine (ortak a,b q05'ten, 3 quantile'a birlikte) DOĞRU yöntem.
  - Bağımsız quantile kalibrasyonu bandı çökertiyor — ELENDİ.
  - N=7 gün yeterli (3/7/14 benzer).
  - Sonuç: station02 %33→%63, station09 %2.7→%53.
  - Entegrasyon regresyon: calibration_data=None → zero-shot ile fark=0.

## ERTELENEN (STAGE-11'de yapılacak)
- **Koşullu otomatik kalibrasyon threshold'u:** (0.15 vs 0.20 tartışması açık).
  - GERÇEK test datasetine göre demo'da ayarlanacak.
  - Eğitim-içi a dağılımı: 8 istasyon |a-1|≤0.164, dkasc outlier 0.544.
  - Doğal kesim 0.164-0.544 arası.
- **Koşullu mantığın kod entegrasyonu:** (predict_intervals'a calibration_data parametresi).
- **Ölçüm belirsizliği argümanı DÜŞÜRÜLDÜ:** (gürültü tabanı 0.59, 0.15'i desteklemedi).

## Açık görevler (sıradaki)
- **STAGE-7:** Optuna optimizasyonu (base learner hiperparametreleri).
- **STAGE-8:** Robustness testleri (sensör arıza senaryoları).
- `experiments/fewshot-calibration` branch'i SAKLA (merge etme, STAGE-11'de kullanılacak).

---

## Sistem
- claude.ai Projects → düşünme, yazım, karar
- Claude Code (terminal) → çalıştırma, git, dosya
- Cowork (bu oturumda kurulum için kullanıldı)
- Makefile komutları: make help
