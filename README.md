Tabii! Aşağıda GitHub README dosyanız için önerilen bir metin bulunmaktadır. Bu README, projenizin temel işlevselliğini, kurulum ve kullanım adımlarını açıklar, ayrıca projenizin sunduğu özellikleri anlatır.

---

# 📊 Analiz Uygulaması

Bu proje, finansal piyasalarda kullanılan teknik analiz araçlarını ve haber analizi ile desteklenmiş karar mekanizmalarını birleştirerek, kullanıcıya yatırım tavsiyesi ve piyasa analizleri sunan bir web uygulamasıdır. Uygulama, döviz kuru, altın fiyatları, Borsa İstanbul gibi finansal enstrümanlar için veri toplar, analiz eder ve kullanıcıya önerilerde bulunur.

🔗 **[Proje GitHub Sayfası](https://github.com/Berke-aras/analizUygulamasi)**

---

## 🚀 Özellikler

- **Teknik Analiz:**
  - **EMA (Exponential Moving Average)**, **RSI (Relative Strength Index)**, **MACD (Moving Average Convergence Divergence)** gibi göstergelerle piyasa analizi.
  - Piyasa volatilitesine göre uygun al/sat kararları.
  
- **Haber Analizi:**
  - Türkçe haber metinlerini analiz ederek, olumsuz veya olumlu haberleri ayıran duygu analizi.
  - Analizler, haber başlıkları ve açıklamalarıyla birlikte sunulur.

- **Gelişmiş Karar Mekanizması:**
  - Teknik ve haber analizlerinin birleştirilerek, al/sat/nötr kararlarının verilmesi.
  - Piyasa trendine ve volatiliteye dayalı karar destek sistemi.

- **Web Arayüzü:**
  - Flask tabanlı bir web uygulaması ile analizleri görsel olarak sunma.
  - Kullanıcı, istediği para birimi veya finansal enstrüman için analiz alabilir.

---

## 🛠️ Teknolojiler

- **Python**: Backend uygulamaları için ana programlama dili.
- **Flask**: Web framework, API ve arayüz oluşturmak için kullanıldı.
- **yfinance**: Finansal veri çekmek için Yahoo Finance API'si.
- **NewsAPI**: Güncel haberleri almak için.
- **NumPy**: Sayısal hesaplamalar ve analizler.
- **Pandas**: Veri işleme ve analizleri için.
- **re**: Metin işleme (regex ile).
- **Matplotlib/Plotly (isteğe bağlı)**: Verilerin görselleştirilmesi (grafik oluşturma).

---

## 🔧 Kurulum

1. **Python ve gerekli kütüphaneleri kurun:**

   Uygulamanın çalışabilmesi için Python 3.7+ ve bazı kütüphanelere ihtiyacınız olacak. Gerekli kütüphaneleri `requirements.txt` dosyasından yükleyebilirsiniz.

   ```bash
   pip install -r requirements.txt
   ```

2. **NewsAPI anahtarınızı alın:**

   [NewsAPI](https://newsapi.org/) üzerinden bir API anahtarı alın ve `NEWS_API_KEY` değişkenine yerleştirin.

3. **Flask Uygulamasını Çalıştırın:**

   Flask uygulamanızı başlatmak için aşağıdaki komutu kullanın:

   ```bash
   python app.py
   ```

   Uygulama, varsayılan olarak `http://localhost:5000` adresinde çalışacaktır.

---

## ⚙️ Kullanım

Uygulamanın ana endpoint'i `/predict`'tir. Buradan, döviz kuru, altın fiyatları gibi finansal enstrümanlar için analiz alabilirsiniz.

### Örnek Kullanım:

**GET İsteği:**
- URL: `http://localhost:5000/predict?currency=dolar`

Bu istek, döviz kuru (USD/TRY) için teknik analiz yapacak ve ilgili haberlerle birlikte yatırım tavsiyesi verecektir.

---

## 🎯 Yapılacaklar

- [ ] Uygulama için kullanıcı dostu bir frontend arayüzü geliştirilmesi.
- [ ] Farklı finansal enstrümanlar için daha geniş veri kümesi entegrasyonu.
- [ ] Duygu analizi ve teknik analizlerin daha sofistike hale getirilmesi.
- [ ] Uygulama optimizasyonları ve hata ayıklamaları.

---

## 💡 Katkı Sağlama

Katkıda bulunmak için aşağıdaki adımları takip edebilirsiniz:

1. Repo'yu forklayın.
2. Yapmak istediğiniz değişiklikleri yeni bir branch üzerinde gerçekleştirin.
3. Pull request (PR) gönderin.

---

## 📜 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

---

## 💬 İletişim

Proje hakkında sorularınız veya önerileriniz varsa, lütfen GitHub Issues bölümünden bize ulaşın veya [berkearas@example.com](mailto:berkearas@example.com) adresinden iletişime geçin.

---

Eğer proje hakkında başka bir şeyler eklemek isterseniz, sorabilirsiniz.
