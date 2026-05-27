# SKILL: Deneysel Branch'lerde Çalışma

## Ne zaman aktif
- experiment/* branch'inde değişiklik yaparken
- Hipotez doğrulamak için izole test koşturduğumuzda
- Ana pipeline'a entegre etmeden önce A/B testi

## Kurallar

**Her deney izole:**
```bash
git checkout -b experiment/{kısa-açıklama}
# Çalışma yap
git commit -m "exp: {hipotez sonucu}"
# Başarılıysa ana dala merge, yoksa branch'i sakla referans olarak
```

**Her deneyin 4 zorunlu çıktısı:**
1. `experiments/{ad}/README.md` — Hipotez, başarı kriterleri
2. `experiments/{ad}/run_experiment.py` — Yeniden koşulabilir kod
3. `experiments/{ad}/results.joblib` — Metrikler
4. `experiments/{ad}/CONCLUSION.md` — Hipotez doğrulandı/çürütüldü, neden

**Başarı kriterleri DENEY ÖNCESİ tanımlanır.**
Post-hoc kriter değişikliği = aldatmaca.

**Karşılaştırma her zaman aynı test setinde.**
A modelini test_v1, B modelini test_v2'de değerlendirme → geçersiz.

**İstatistiksel test:**
- İki model karşılaştırma → Diebold-Mariano testi
- p-value < 0.05 → istatistiksel olarak anlamlı
- Sadece point estimate karşılaştırma yetersiz

## Yapma
- Ana dalda direkt deneme
- Hipotezi sonradan değiştirme
- "Sadece bu sefer" istisnası
- Failed deneyleri silme (CONCLUSION.md ile sakla, bilgi kaybı olmaz)
