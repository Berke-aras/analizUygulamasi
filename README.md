Trading Assistant Uygulaması

Bu uygulama, Flask tabanlı bir web servisi olarak çalışır ve hem teknik analiz hem de haber analizi yaparak döviz, altın veya Borsa İstanbul gibi finansal enstrümanlar için al/sat/nötr tavsiyeleri üretir.
İçerik

    Özellikler

    Teknolojiler

    Kurulum

    Örnek Çıktı

Özellikler

    Haber Analizi: Haber API’si (NewsAPI) kullanılarak çekilen haber verileri, Türkçe stop-word filtresi ve duygu analizi algoritmaları ile işlenir.

    Teknik Analiz:

        EMA (Üssel Hareketli Ortalama)

        RSI (Göreceli Güç Endeksi)

        MACD (Hareketli Ortalama Yakınsaması/Uzaklaşması)

        Ek göstergeler (momentum, volatilite) ile harmanlanmış analiz.

    Karar Mekanizması: Teknik ve haber skorlarına dayanarak “AL”, “SAT” veya “NÖTR” gibi işlem önerileri sunar.

    Grafik Verisi: Son 60 güne ait fiyat grafiği verileri JSON formatında sunulur.

    Haber Özeti: İlk 10 haber, kaynak bilgileriyle birlikte özetlenir.

Teknolojiler

    Python

    Flask

    yfinance

    Pandas & NumPy

    NewsAPI

    Regex (re modülü)

Kurulum

    Repository’yi Klonlayın:

git clone https://github.com/kullaniciadi/trading-assistant.git
cd trading-assistant

Gerekli Python Paketlerini Yükleyin:

pip install -r requirements.txt

Eğer requirements.txt dosyanız yoksa aşağıdaki paketleri yükleyebilirsiniz:

    pip install flask requests yfinance numpy pandas

    API Anahtarını Ayarlayın:

    Uygulama içindeki NEWS_API_KEY değişkenine NewsAPI üzerinden aldığınız API anahtarınızı girin.

Kullanım

    Uygulamayı Başlatın:

python app.py

Web Tarayıcınızdan Erişin:

Uygulama, varsayılan olarak http://127.0.0.1:5000 adresinde çalışacaktır.



Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen bir pull request gönderin veya issue açın. Her türlü öneri ve geri bildirim değerlidir.
Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.
