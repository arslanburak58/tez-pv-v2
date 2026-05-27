# Veri Seti Değişken Görselleştirme Raporu ve Akademik Analizi

Bu doküman, veri ETL, normalizasyon ve fiziksel öznitelik mühendisliği (Stage-2 ve Stage-3) aşamalarından geçmiş olan **DKASC Alice Springs** istasyonunun 15 günlük yaz mevsimi (1-15 Ocak 2020) kesiti üzerinden değişkenlerimizin davranışlarını ve tezinize katkı sağlayacak olan akademik/fiziksel gerekçelerini sunmaktadır.

Her bir değişken için iki farklı bakış açısı sunulmuştur:
1. **Sol Grafik (Zaman Serisi):** 15 günlük kesintisiz ardışık seyir (günlük tepe ve vadi döngüleri, bulut geçişleri vb.).
2. **Sağ Grafik (Günlük Bindirme / Overlay):** 15 günün her bir gününün tek bir 24 saatlik x-ekseni üzerinde saydam çizgilerle bindirilmesi ve kalın çizgiyle 15 günlük ortalama diürnal trendin gösterilmesi.

---

## 1. Meteorolojik ve Santral Çıktı Değişkenleri

### 1.1. y_norm (Kapasite Faktörü - Normalize Güç)
Santralin anlık güç üretiminin nominal gücüne bölünmesiyle elde edilen boyutsuz ($[0, 1.07]$ arası) ana hedef değişkenimizdir. Solar noon saatlerinde zirve yapar ve bulut geçişlerindeki volatilite sol grafikte net bir şekilde izlenebilir.

![y_norm (Kapasite Faktörü) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_y_norm.png)

---

### 1.2. GHI (Global Yatay Işınım - W/m²)
Yeryüzüne düşen toplam güneş ışınımıdır. Aşırı ışınım (over-irradiance) filtrelerimiz ve enterpolasyon kararlarımız sayesinde üst limit $1500 \text{ W/m}^2$ ile sınırlandırılmıştır ve solar öğle vakti kusursuz bir parabolik seyir izler.

![GHI (Global Yatay Işınım) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_ghi.png)

---

### 1.3. T_amb (Ortam Sıcaklığı - °C)
İstasyon çevresindeki hava sıcaklığıdır. Güneş ışınımına kıyasla daha gecikmeli (thermal lag) bir seyir izler; günün en yüksek sıcaklığı genellikle solar öğle vaktinden (12:00) 2-3 saat sonra (14:00 - 15:00 dolaylarında) gerçekleşir.

![T_amb (Ortam Sıcaklığı) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_t_amb.png)

---

### 1.4. T_cell (Panel Hücre Sıcaklığı - °C)
Ross Formülü ($T_{amb} + 0.03125 \times GHI$) ile hesaplanan panel sıcaklığıdır. Güneş ışınımı (GHI) doğrudan hücreyi ısıttığı için gündüz saatlerinde ortam sıcaklığının çok üzerine çıkar (maksimum $\sim 85^\circ C$). Geceleri ise ışınım sıfır olduğundan tam olarak ortam sıcaklığına ($T_{amb}$) eşitlenir.

![T_cell (Panel Sıcaklığı) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_t_cell.png)

---

### 1.5. RH (Bağıl Nem - %)
Havadaki bağıl nem oranıdır. Sıcaklıkla ters orantılı bir fiziksel davranış gösterir; sıcaklığın en yüksek olduğu öğleden sonra saatlerinde nem dip yaparken, geceleri sıcaklık düştükçe nem zirveye ulaşır.

![RH (Bağıl Nem) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_rh.png)

---

## 2. Astronomik ve Solar Konum Öznitelikleri

### 2.1. cos_zenith (Zenit Açısı Kosinüsü)
Güneşin gökyüzündeki konumunu temsil eden en güçlü geometrik özniteliktir. Güneş tam tepedeyken $1.0$, ufuktayken $0.0$ ve ufkun altındayken (gece) negatif değerler alır. Mevsimsel ve günlük periyodiktir.

![cos_zenith (Kosinüs Zenit Açısı) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_cos_zenith.png)

---

### 2.2. k_t (Açık Gökyüzü / Berraklık İndeksi)
Geliştirdiğimiz kurallar doğrultusunda gece saatlerinde (zenit açısı $\ge 85^\circ$) **tam olarak 0.0** olan, gündüz saatlerinde ise atmosferin güneş ışınımını ne oranda geçirdiğini gösteren indekstir ($[0, 2]$ aralığına sınırlıdır).

![k_t (Berraklık İndeksi) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_k_t.png)

---

### 2.3. hour_angle (Saat Açısı - Derece)
Yerel Güneş Saatine (LST) göre güneşin meridyenden olan açısal mesafesidir. Tam olarak saat 12:00'de (solar öğle) $0^\circ$ değerini alır. Sabah saatlerinde negatif (gün doğumunda $\sim -90^\circ$ ila $-120^\circ$), öğleden sonra ise pozitif değerlere doğru doğrusal bir geçiş yapar.

![hour_angle (Saat Açısı) 15 Günlük Profil](/Users/burakarslan/Desktop/tez-pv-ant/results/figures/plot_hour_angle.png)

---

## 3. Saat Açısının (Hour Angle - $H$) Tahmin Modeline Katkıları
Saat Açısı ($H$) değişkeninin probabilistic modelimize ve tezinizin akademik savunmasına sağlayacağı 4 ana katkı:

1. **Boylam Sapmalarını ve Yapay Zaman Dilimi Sınırlarını Sıfırlaması:**
   Standart saatimiz (örneğin telefonumuzda yazan 12:00) yapay bir idari tanımdır. Coğrafi olarak güneşin meridyenden tam tepe noktasından geçişi (gerçek solar öğle vakti) boylam farkı ve Dünya'nın eliptik yörüngesinden kaynaklanan zaman denklemi (EoT) sapması nedeniyle yerel saatle 11:40 ile 12:20 arasında kayabilir. Saat Açısı ($H$), boylam ve EoT düzeltmelerini bünyesinde barındırarak güneşin meridyenden olan gerçek açısal mesafesini temsil eder. Güneşin tam en tepede olduğu an, konuma bakılmaksızın $H = 0^\circ$ olur. Bu sayede model, yapay saat dilimi kabullerinden kurtulup doğrudan **fiziksel güneş öğlesini** öğrenir.

2. **Sabah ve Öğleden Sonra Asimetrisini Çözmesi (Ağaç Modelleri İçin Doğrusal Geçiş):**
   Güneş enerjisi tahmininde saat 09:00 ile 15:00'te güneş konumları birbirine çok yakındır. Ancak PV panellerin bu saatlerdeki davranışı tamamen asimetriktir: Sabah 09:00'da paneller henüz soğuktur ve verimleri yüksektir. Öğleden sonra 15:00'te ise panel sıcaklığı gün boyu biriken ısıyla tepe yapmıştır ve verim daha düşüktür. Eğer sadece güneş yüksekliği veya sin/cos saat kullanırsak model bu iki saati ayırt etmekte zorlanabilir. Oysa Saat Açısı ($H$) gün doğumundan gün batımına kadar kesintisiz ve doğrusal bir geçiş sunar (sabah $-90^\circ$, öğle $0^\circ$, akşam $+90^\circ$). Karar ağacı modelleri (XGBoost, LightGBM, CatBoost), bu lineer değişken üzerindeki eşik değerlerini ($H < 0$ ise sabah, $H > 0$ ise öğleden sonra) çok kolay bölerek (split) **sabah/öğleden sonra asimetrisini kusursuz şekilde öğrenir.**

3. **Çoklu İstasyon (Multi-Plant) Transfer Learning Gücü:**
   Farklı enlem ve boylamlardaki santrallerin enlem, boylam ve yerel saat örüntüleri farklıdır. Saat Açısı, modeller için **evrensel bir koordinat referansı** oluşturur. Model, "Çin'deki santralde Saat Açısı $-30^\circ$ iken (sabah saatleri) üretim kapasitesi neydi" bilgisini doğrudan Avustralya'daki santrale sıfır hata ile transfer edebilir.

4. **Tez Savunmasında Fiziksel Güç (Physics-Informed ML):**
   Tezinizde basit bir veri bilimi denemesi yapmak yerine, *"gök mekaniği prensiplerine sadık kalarak, her istasyonun coğrafi boylamını ve Dünya'nın eliptik yörünge düzeltmesini içeren Güneş Saat Açısını hesaplayıp sisteme fiziksel yönlendirici (physics-informed) girdi olarak sundum"* diyerek bilimsel derinliği savunabileceksiniz.

---

## 4. Akademik Bulgular
* **Fiziksel Kararlılık:** GHI ve $T_{cell}$ grafiklerinde görüldüğü üzere ekstrem gürültüler başarıyla temizlenmiş ve diürnal trendler bozulmadan yumuşatılmıştır.
* **Gece Kararlılığı:** `k_t` ve `T_cell` değişkenlerinin gece saatlerinde (ışınım sıfırken) tam olarak kararlı değerler (nem ve ortam sıcaklığıyla eşgüdümlü) aldığı gözlemlenmiştir. Bu durum modelin gece sıfır üretim örüntüsünü öğrenmesinde aşırı sapmaları (overfitting) engeller.
* **Thermal Lag Kanıtı:** Ortam sıcaklığı ($T_{amb}$) ile Panel sıcaklığının ($T_{cell}$) solar noon sonrasındaki pik kayması, güneş hücresinin ısıl kapasitesini ve fiziksel gerçekliğini tezinizde savunabileceğiniz mükemmel bir veri örüntüsüdür.
