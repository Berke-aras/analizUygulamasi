import re
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import string
from flask import Flask, request, jsonify, render_template, abort
import traceback

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # JSON sıralamasını kapat
NEWS_API_KEY = "ef2f45c56b17491b9c8a10ea947d5e40"

# Küçük kapsamlı Türkçe stop-word listesi
TURKISH_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey", "biz",
    "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi", "hem", 
    "hep", "hepsi", "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", 
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o", "sanki", "şey", "siz", 
    "şu", "tüm", "ve", "veya", "ya", "yani"
}

# 1. Geliştirilmiş Haber Analizi Fonksiyonları
def fetch_news(query):
    """
    Haber API'sinden ham veriyi alır ve temizler.
    """
    url = f"https://newsapi.org/v2/everything?q={query}&language=tr&pageSize=20&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        clean_articles = []
        for article in articles:
            item = {
                'title': article.get('title', '').strip(),
                'description': article.get('description', '').strip(),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', '#'),
                'publishedAt': article.get('publishedAt', '')
            }
            if item['title'] or item['description']:
                clean_articles.append(item)
        
        return clean_articles
    
    except Exception as e:
        print(f"Haber API Hatası: {str(e)}")
        return []

def preprocess_text(text):
    """
    Metni küçük harfe çevirir, noktalama işaretlerini kaldırır ve Türkçe stop-word'leri temizler.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    filtered_words = [word for word in words if word not in TURKISH_STOPWORDS]
    return filtered_words

def analyze_news(news_text):
    """
    Gelişmiş haber analizi:
    - Gelişmiş tokenizasyon ve stop-word temizliği
    - Basit negatif kontrol ile duygu skorlaması
    """
    # Pozitif ve negatif duygu kelimeleri
    positive = {'yükseliş','artış','kazanç','rekor','güçlü','olumlu','başarı','fırladı','patlama','zirve','iyi','gelişme'}
    negative = {'düşüş','kriz','zarar','kayıp','çöküş','tartışma','düşecek','vurgun','kaybetti','iflas','durgunluk','risk'}
    negation_words = {'değil','ama','ancak','rağmen','olumsuz'}

    sentences = re.split(r'[.!?]+', news_text)
    total_score = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        words = preprocess_text(sentence)
        sentence_score = 0
        negation = False
        
        for word in words:
            if word in negation_words:
                negation = not negation
                continue
            if word in positive:
                score = 2 if not negation else -2
                sentence_score += score
            elif word in negative:
                score = -2 if not negation else 2
                sentence_score += score
        
        if words:
            total_score += sentence_score / len(words)
    
    # Toplam skoru ölçeklendir ve sınırla
    scaled_score = np.clip(total_score * 10, -5, 5)
    return scaled_score

# 2. Gelişmiş Teknik Analiz Fonksiyonları
def calculate_ema(prices, period):
    """
    Güvenli EMA Hesaplama: Yeterli veri yoksa None döndürür.
    """
    try:
        if not prices or len(prices) < period:
            return None
        valid_prices = prices[-3*period:]
        series = pd.Series(valid_prices)
        return series.ewm(span=period, adjust=False).mean().iloc[-1]
    except Exception as e:
        print(f"EMA {period} Hata: {str(e)}")
        return None

def calculate_rsi(prices, period=14):
    """
    Düzeltilmiş RSI Hesaplama.
    """
    if len(prices) < period+1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100/(1 + rs))

def calculate_macd(prices):
    """
    Optimize MACD Hesaplama.
    """
    try:
        if len(prices) < 26:
            return 0
        
        ema12 = calculate_ema(prices, 12)
        ema26 = calculate_ema(prices, 26)
        
        if ema12 is None or ema26 is None:
            return 0
        
        macd_line = ema12 - ema26
        macd_values = []
        
        for i in range(26, len(prices)):
            ema12_part = calculate_ema(prices[:i+1], 12)
            ema26_part = calculate_ema(prices[:i+1], 26)
            if ema12_part is not None and ema26_part is not None:
                macd_values.append(ema12_part - ema26_part)
        
        if len(macd_values) < 9:
            return 0
        
        signal_line = calculate_ema(macd_values, 9) or 0
        return macd_line - signal_line
    
    except Exception as e:
        print(f"MACD Hesaplama Hatası: {str(e)}")
        return 0

def analyze_chart(price_history):
    """
    Teknik göstergeler kullanılarak toplam teknik skoru hesaplar.
    """
    try:
        ema50 = calculate_ema(price_history, 50) or 0
        ema200 = calculate_ema(price_history, 200) or 1  # Sıfıra bölmeyi önlemek için
        
        indicators = {
            'rsi': calculate_rsi(price_history),
            'macd': calculate_macd(price_history),
            '50_200_ema': ema50 / ema200 if ema200 != 0 else 1,
            'momentum': (price_history[-1]/price_history[-14] - 1)*100 if len(price_history) >= 14 else 0,
            'volatility': np.std(price_history[-30:]) / np.mean(price_history[-30:]) if len(price_history) >= 30 else 0
        }
        
        # RSI için basit yorum
        rsi_score = -2 if indicators['rsi'] > 70 else (2 if indicators['rsi'] < 30 else 0)
        macd_score = indicators['macd'] * 0.1
        ema_score = (indicators['50_200_ema'] - 1) * 1000
        
        total_tech_score = (
            rsi_score * 0.4 +
            macd_score * 0.3 +
            ema_score * 0.2 +
            indicators['momentum'] * 0.1
        )
        
        return total_tech_score
    
    except Exception as e:
        print(f"Chart analiz hatası: {str(e)}")
        return 0

# 3. Gelişmiş Karar Mekanizması
def decide_action(technical_score, news_score, price_history):
    """
    Teknik ve haber skorlarına göre al/sat/nötr karar verir.
    """
    try:
        if len(price_history) < 30:
            return "YETERSİZ VERİ"
        
        last_30 = price_history[-30:]
        volatility = np.std(last_30) / np.mean(last_30) if np.mean(last_30) != 0 else 0
        
        thresholds = {
            'strong': 5 + (volatility * 300),
            'moderate': 2 + (volatility * 150)
        }
        
        total_score = technical_score * 0.65 + news_score * 0.35
        momentum = (price_history[-1] / price_history[-14] - 1) * 100 if len(price_history) >= 14 else 0
        
        if total_score > thresholds['strong']:
            return "GÜÇLÜ AL" if momentum > -5 else "AL"
        elif total_score > thresholds['moderate']:
            return "AL"
        elif total_score < -thresholds['strong']:
            return "ACİL SAT" if momentum < 5 else "SAT"
        elif total_score < -thresholds['moderate']:
            return "SAT"
        else:
            if momentum > 7:
                return "AL (Momentum)"
            elif momentum < -7:
                return "SAT (Momentum)"
            else:
                return "NÖTR"
    
    except Exception as e:
        print(f"Karar Mekanizması Hatası: {str(e)}")
        return "HATA"

# Web uygulaması için rotalar
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    currency = request.args.get("currency", "dolar").lower().strip()
    
    ticker_map = {
        'dolar': ('USDTRY=X', 'dolar kur'),
        'euro': ('EURTRY=X', 'euro kur'),
        'altın': ('GC=F', 'altın fiyatları'),
        'bist': ('XU100.IS', 'borsa istanbul')
    }
    
    try:
        if currency not in ticker_map:
            raise ValueError("Desteklenmeyen para birimi")
            
        ticker, query = ticker_map[currency]
        
        # Fiyat verisi çekme
        price_data = yf.download(
            ticker,
            period='2y',
            interval='1d',
            auto_adjust=False,
            actions=False,
            progress=False
        )
        
        if price_data.empty:
            raise ValueError("Boş veri seti")
            
        # Fiyat sütunu kontrolü
        if 'Close' not in price_data.columns:
            if 'Adj Close' in price_data.columns:
                price_data['Close'] = price_data['Adj Close'].copy()
            else:
                raise ValueError("Fiyat sütunu bulunamadı")
        
        price_history = price_data['Close'].squeeze().dropna().tolist()
        if len(price_history) < 30:
            raise ValueError(f"Yetersiz veri: {len(price_history)}/30 gün")
        
        # Haber verisi çekme ve birleştirme
        news_articles = fetch_news(query)
        news_text = " ".join(
            f"{a['title']} {a['description']}" 
            for a in news_articles if a['title'] or a['description']
        )
        
        # Analizler
        tech_score = analyze_chart(price_history)
        news_score = analyze_news(news_text)
        action = decide_action(tech_score, news_score, price_history)
        
        # Grafik verisi hazırlama (son 60 gün)
        chart_data = {
            'dates': price_data.index.strftime('%Y-%m-%d').tolist()[-60:],
            'prices': price_history[-60:]
        }
        
        # Haberlerin formatlanması (ilk 10 haber)
        formatted_news = []
        for article in news_articles[:10]:
            formatted_news.append({
                'title': article.get('title', 'Başlıksız'),
                'source': article.get('source', 'Bilinmeyen Kaynak'),
                'publishedAt': article.get('publishedAt', 'Tarih Yok'),
                'url': article.get('url', '#')
            })
        
        result = {
            'para_birimi': currency.upper(),
            'güncel_fiyat': round(price_history[-1], 2),
            'teknik_skor': round(tech_score, 2),
            'haber_skoru': round(news_score, 2),
            'tavsiye': action,
            'göstergeler': {
                '7_gunluk': f"{(price_history[-1] / price_history[-7] - 1) * 100:.2f}%" if len(price_history) >= 7 else "N/A",
                '30_gunluk': f"{(price_history[-1] / price_history[-30] - 1) * 100:.2f}%" if len(price_history) >= 30 else "N/A"
            },
            'chart_data': chart_data,
            'haberler': formatted_news
        }
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Hata: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "hata": str(e),
            "detay": traceback.format_exc() if app.debug else None
        }), 500
        
if __name__ == '__main__':
    app.run(debug=False)
