# v1'den Öğrendiklerimiz — Hata, Sebep, Çözüm

Bu dosya v2'de tekrarlanmaması gereken hataları tutar.

---

## Ders 1: Kapasite Normalizasyonu Şart

**Hata:** v1'de mutlak kW değerleriyle eğitildi (DKASC ~150 kW).
**Sonuç:** Model "150 kW tavan" öğrendi. PVOD 17 MW santral'e uygulandığında q50 ≈ 0.009 (normalize), oysa actual 0.7.
**Çözüm (v2):** Her şey kapasite faktörüyle [0,1] aralığında eğitilir.

---

## Ders 2: Gradient Boosting Pinball Yaklaşımı q01 Collapse Yapar

**Hata:** XGBoost custom pinball objective (gradient/hessian yaklaşımı) q01'i 0'a doğru çekiyor.
**Mekanizma:** Gradient = `(y_pred >= y_true) ? q : (q-1)`. q=0.1 için gradient ya -0.1 ya +0.9. Loss "küçük tahmin → küçük ceza" stratejisini öğreniyor.
**Çözüm (v2):** Meta-katmanda sklearn QuantileRegressor LP solver gerçek pinball'u tam çözer. q01 collapse'i mathematically imkansızlaştırır.

---

## Ders 3: CQR Bir Kludge

**Hata:** v1'de coverage 60% kalınca CQR k=1.7 post-hoc uygulandı, 81%'e çıkarıldı.
**Sorun:** CQR semptomu örtüyor, sebebi çözmüyor. k değeri başka veri setinde geçersiz.
**Çözüm (v2):** Sebebi (Ders 2) çöz, CQR'a gerek kalmasın.

---

## Ders 4: Erken EDA = Geç Bug

**Hata:** PVOD power'ın MW birimde olduğunu STAGE-11'e gelene kadar fark etmedik.
**Sebep:** `data_dictionary.md` STAGE-2'de yazılmıştı ama unit doğrulaması yapılmamış.
**Çözüm (v2):** STAGE-2'de her sütun için `min, max, median, units` kontrolü, otomatik test ekle.

---

## Ders 5: Streamlit Cache Sinsidir

**Hata:** Kod değişiyor ama Streamlit eski model state'i tutuyor. Debug saatlerce yanıltıcı.
**Çözüm (v2):** Geliştirme sırasında `--server.runOnSave=true` + `st.cache_data.clear()` butonu.

---

## Ders 6: Python 3.13'te dataset.joblib Açılmıyor

**Hata:** v1 dataset 3.11/3.12 ile kaydedildi, 3.13'te `StringDtype` uyumsuzluğu.
**Çözüm (v2):** Python sürümünü .python-version dosyasına sabitle (3.11.x). Tüm joblib dosyaları aynı sürümde.

---

## Ders 7: gunce.md GitHub Cache Sorunu

**Hata:** raw.githubusercontent.com cache eski versiyonu sunuyor, Projects sync gecikiyor.
**Çözüm (v2):** Her oturum başında kullanıcı gunce.md içeriğini kendisi yapıştırır. Fetch'e güven yok.

---

## Ders 8: "Aşama Atlatma" Tuzağı

**Hata:** Bir aşamada problem çıkınca "şimdilik geç, sonra dönerim" → asla geri dönülmedi, sorun büyüdü.
**Çözüm (v2):** Her aşamanın "Tamamlandı kriteri" var, sağlanmadan sonraki başlamaz.

---

## Ders 9: Yapay Kapsama Oranı (All-Hours Coverage) Tuzağı

**Hata:** Gece/sıfır-üretim saatlerini (%100 doğru tahmin edilen) değerlendirmeye dahil ederek elde edilen %80.19'luk all-hours coverage oranını yanıltıcı bir nihai başarı ölçütü olarak çerçevelemek.
**Sonuç:** Modelin aktif gündüz/üretim saatlerindeki gerçek fiziksel kapsama performansının yetersizliği (%65.20) maskelenmekte ve tez metninde akademik olarak zayıf/tehlikeli bir argüman oluşmaktadır.
**Çözüm (v2):** 
1. Akademik dürüstlük ve teknik doğruluk açısından nihai performans metriği olarak **yalnızca aktif gündüz/üretim saatlerindeki (daylight/production hours) kapsama oranını** hedef almak.
2. STAGE-7 Optuna optimizasyonunda hiperparametreleri sadece pinball loss'a göre değil, **aktif üretim saatlerindeki kapsama oranını %80'e yaklaştıracak biçimde** optimize etmek.

