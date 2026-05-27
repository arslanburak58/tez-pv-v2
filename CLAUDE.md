# Proje Bağlamı — tez-pv-v2

Ben Burak Arslan, Sivas Cumhuriyet Üniversitesi YL öğrencisi.
Bu proje yüksek lisans tezimin uygulama kısmıdır.

## Tez Başlığı
"Fotovoltaik Sistemlerde Sensör Kayıplarına Dayanıklı Olasılıksal Güç Tahmini"

## Mimari Özet
- Stacked ensemble: 3 algoritma (XGBoost, LightGBM, CatBoost) × 3 quantile (q=0.1, 0.5, 0.9)
- Meta-öğrenici: sklearn QuantileRegressor (HiGHS LP) — proposal'da Ridge yazıyor ama LP-tabanlı QR teorik olarak daha doğru
- Hedef değişken: y = power_kW / capacity_kW ∈ [0,1] (multi-plant transfer için)
- Eğitim verisi: DKASC Alice Springs + PVOD v1.0 (8 istasyon, 2 holdout)

## Donanım
- MacBook Air M4 (Apple Silicon)
- PyTorch için MPS backend (LSTM/TFT baseline'larında)
- Ağır işlerde Google Colab T4

## Stage Sistemi
- Detay: `docs/tez_workflow.md`
- Aktif durum: `docs/gunce.md` (her oturum başında oku)
- Tamamlanma kayıtları: `docs/stage_log.md`
- Yöntemsel kararlar: `docs/methodology_decisions.md`
- v1 derslerimiz: `docs/lessons_learned.md`

## Skill Dosyaları (bağlama göre devreye al)
- `docs/skills/skill_code_helper.md` — Python kodu, debug, pipeline
- `docs/skills/skill_lit_synth.md` — Literatür sentezi
- `docs/skills/skill_methods_writer.md` — Yöntem bölümü yazımı
- `docs/skills/skill_reviewer.md` — Eleştirel inceleme
- `docs/skills/skill_experiment_runner.md` — Deneysel branch'lerde çalışırken

## Genel Kurallar
- Türkçe yanıt. Teknik terimler İngilizce kalır.
- Tip ipuçları zorunlu (Python 3.11+).
- Reprodüksiyon: `random.seed(42)`, `np.random.seed(42)`.
- M4 donanım kısıtı; ağır eğitimde Colab öner.
- Veri sızıntısı kontrolü: scaler/imputer/encoder sadece train'de fit.
- Walk-forward CV (TimeSeriesSplit, gap=24h).
- Joblib model serileştirme (pickle değil).

## Yapma
- Atfa atlama: Eski projede yaptık diye direkt geçme. v2'de her stage yeniden.
- Acele etme: "süre önemli değil, sağlam temele otursun" — kullanıcının kuralı.
- CQR önerme: LP doğru çözüyor, post-hoc kalibrasyon gereksiz.
- Mutlak kW ile eğitim: Her şey normalize olacak.
