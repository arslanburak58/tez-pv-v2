# SKILL: Tez Yöntem Bölümü Yazımı

## Ne zaman aktif
Tez yöntem paragrafı, formül anlatımı, algoritma tarifi yazılırken.

## Kurallar

**Üslup:** Türkçe akademik, edilgen yapı uygun.

**Her alt bölüm için:**
1. Tek cümle amaç ("Bu adımda... gerçekleştirilir.")
2. Matematiksel formülasyon (varsa)
3. İmplementasyon detayı (kütüphane, fonksiyon, sürüm)
4. Hiperparametre veya tasarım seçimi (literatür gerekçesiyle)
5. Sınırlılık veya varsayım (açık belirt)

**Atıflar:** APA, inline (Yazar, Yıl).

**Reprodüksiyon vurgusu:** Her algoritma için seed, sürüm, donanım.

## v2-Spesifik Terminoloji

- "Kapasite normalizasyonu" — power/capacity oranı
- "Quantile regresyon" (İngilizce, çevirme)
- "Stacked ensemble" / "Yığılmış topluluk" (ilk kullanımda iki dilli)
- "Meta-öğrenici" (meta-learner değil)
- "Sensör kaybına dayanıklılık" (robustness değil)
- "Eksik veri bayrakları" (missingness indicators)
- "Açıklık indeksi" (k_t, clear-sky index)
- "Açık-gökyüzü modeli" (clear-sky model)
- "Walk-forward çapraz doğrulama" (zaman serisi CV)

## Kapasite Normalizasyonu Cümlesi (örnek)

> Çoklu santral genelleme yetisi için hedef değişken kapasite-normalize edilmiş güç çıkışı olarak tanımlanır:
> 
> *y_norm = P_kW / C_kW*
> 
> Burada P_kW ölçülen güç, C_kW santralin nominal kapasitesidir. Bu dönüşüm modelin farklı ölçeklerdeki santraller üzerinde tek bir parametre setiyle eğitilmesini sağlar (Karar 1, methodology_decisions.md).

## QR Meta-Öğrenici Cümlesi (örnek)

> Meta-öğrenici olarak lineer quantile regresyon (Koenker ve Bassett, 1978) seçilmiştir. Bu yaklaşımın klasik Ridge regresyondan farkı, MSE yerine pinball loss fonksiyonunu doğrudan minimize etmesidir:
> 
> *β̂_q = argmin Σ_i ρ_q(y_i - x_i^T β)*
> 
> burada ρ_q(u) = u·(q - 𝟙{u<0}). Bu LP problemi sklearn'ün `QuantileRegressor` sınıfı aracılığıyla HiGHS solver ile çözülmüştür (Huangfu ve Hall, 2018).

## Yapma
- "burada şunu yaptık" tarzı günlük üslup
- Atıfsız metodolojik karar
- "yenilikçi", "üstün", "etkileyici" sıfatları
- Belirsiz cümleler ("önemli bir teknik kullanıldı")
- Eksik koşul ("model eğitildi" → hangi setle, ne süre, hangi durdurma?)
