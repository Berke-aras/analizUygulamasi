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
    return float(100 - (100/(1 + rs)))

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
            'rsi': float(calculate_rsi(price_history)),
            'macd': float(calculate_macd(price_history)),
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

# app.py dosyasındaki predict fonksiyonunun güncellenmiş hali

@app.route('/predict', methods=['GET'])
def predict():
    currency = request.args.get("currency", "dolar").lower().strip()
    
    # Özel hisse senedi seçilmiş mi kontrol et
    
    ticker_map = {
        'dolar': ('USDTRY=X', 'dolar kur'),
        'euro': ('EURTRY=X', 'euro kur'),
        'altın': ('GC=F', 'altın fiyatları'),
        'bist': ('XU100.IS', 'borsa istanbul'),
        'gümüş': ('SI=F', 'gümüş fiyatları'),
        'platin': ('PL=F', 'platin fiyatları'),
        'bakır': ('HG=F', 'bakır fiyatları'),
        'petrol': ('CL=F', 'petrol fiyatları'),
        'nasdaq': ('^IXIC', 'nasdaq borsa'),
        'sp500': ('^GSPC', 'sp500 endeks'),
        'dow': ('^DJI', 'dow jones')
    }
    
    is_custom_stock = not currency in ticker_map and currency != "custom"
    
    try:
        # Özel hisse senedi mi yoksa önceden tanımlı bir para birimi mi?
        if is_custom_stock:
            ticker, query = currency.upper(), f"{currency} stock news"
        else:
            if currency not in ticker_map:
                raise ValueError("Desteklenmeyen para birimi veya finansal enstrüman")
                
            ticker, query = ticker_map[currency]
    except Exception as e:
        app.logger.error(f"Özel hisse senedi kontrol hatası: {str(e)}")
        return jsonify({"hata": "Özel hisse senedi kontrol hatası"}), 400
    
    try:
        # Özel hisse senedi mi yoksa önceden tanımlı bir para birimi mi?
        if is_custom_stock:
            ticker, query = currency.upper(), f"{currency} stock news"
        else:
            if currency not in ticker_map:
                raise ValueError("Desteklenmeyen para birimi veya finansal enstrüman")
                
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
        
        price_history = price_data['Close'].squeeze().dropna().to_list()
        if len(price_history) < 30:
            raise ValueError(f"Yetersiz veri: {len(price_history)}/30 gün")
        
        # Yüksek ve düşük değerleri al (teknik analizler için)
        high_history = price_data['High'].dropna().values.tolist() if 'High' in price_data.columns else None
        low_history = price_data['Low'].dropna().values.tolist() if 'Low' in price_data.columns else None
        
        # Haber verisi çekme ve birleştirme
        news_articles = fetch_news(query)
        news_text = " ".join(
            f"{a['title']} {a['description']}" 
            for a in news_articles if a['title'] or a['description']
        )
        
        # Analizler
        tech_score = analyze_chart(price_history)
        news_score = analyze_news(news_text)
        
        # Gelişmiş analizler
        supports, resistances = detect_support_resistance(price_history)
        sma, upper_band, lower_band = calculate_bollinger_bands(price_history)
        stoch_k, stoch_d = calculate_stochastic_oscillator(price_history, high_history, low_history)
        rsi_value = calculate_rsi(price_history)
        macd_value = calculate_macd(price_history)
        market_condition = identify_market_condition(price_history)
        
        # Fırsat analizi
        opportunity_analysis = analyze_opportunity(price_history, tech_score, news_score)
        
        # Temel tavsiye (geriye uyumluluk için)
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
        
        # Para birimi belirleme
        if is_custom_stock:
            # Yahoo Finance'den sembol bilgisini çek
            ticker_info = yf.Ticker(ticker).info
            display_currency = ticker_info.get('currency', 'USD').upper()
            display_name = ticker_info.get('shortName', ticker)
        else:
            display_currency = currency.upper()
            display_name = ticker_map[currency][1].upper() if currency in ticker_map else currency.upper()
        
        result = {
            'para_birimi': display_currency,
            'sembol': ticker,
            'isim': display_name,
            'güncel_fiyat': float(round(price_history[-1], 2)),  # numpy tipi varsa çevir
            'teknik_skor': float(round(tech_score, 2)),
            'haber_skoru': float(round(news_score, 2)),
            'tavsiye': action,
            'piyasa_durumu': market_condition,
            'göstergeler': {
                '7_gunluk': f"{(price_history[-1] / price_history[-7] - 1) * 100:.2f}%" if len(price_history) >= 7 else "N/A",
                '30_gunluk': f"{(price_history[-1] / price_history[-30] - 1) * 100:.2f}%" if len(price_history) >= 30 else "N/A"
            },
            'detaylı_teknik': {
                'rsi': rsi_value,
                'macd': macd_value,
                'stochastic': {'k': stoch_k, 'd': stoch_d},
                'supports': supports,
                'resistances': resistances,
                'bollinger': {
                    'sma': sma,
                    'upper': upper_band,
                    'lower': lower_band
                }
            },
            'fırsat_analizi': opportunity_analysis,
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

# app.py içine eklenecek hisse senedi arama fonksiyonu
def search_stock(query):
    """
    Verilen sorguya göre hisse senetlerini arar ve bulur
    """
    try:
        tickers = yf.Tickers(query)
        if not tickers.tickers:
            return None, "Hisse senedi bulunamadı"
            
        # İlk eşleşen hisse senedini al
        ticker = list(tickers.tickers.values())[0]
        info = ticker.info
        
        # Temel bilgileri döndür
        return {
            'symbol': info.get('symbol', query),
            'name': info.get('shortName', 'Bilinmeyen'),
            'sector': info.get('sector', 'Bilinmeyen Sektör'),
            'price': info.get('currentPrice', 0),
            'currency': info.get('currency', 'USD'),
            'marketCap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0)
            # Remove the ticker object from the response
        }, None
    except Exception as e:
        return None, f"Hisse senedi arama hatası: {str(e)}"
        
        

# app.py içine eklenecek yeni endpoint
@app.route('/search', methods=['GET'])
def search():
    """
    Hisse senedi arama endpoint'i
    """
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"hata": "Arama sorgusu gerekli"}), 400
        
    result, error = search_stock(query)
    if error:
        return jsonify({"hata": error}), 404
        
    return jsonify(result)


# app.py içine eklenecek gelişmiş teknik analiz fonksiyonları

def detect_support_resistance(price_history, window=10):
    """
    Basit bir destek ve direnç seviyesi tespit algoritması.
    """
    supports = []
    resistances = []
    
    if len(price_history) < window*2:
        return [], []
        
    # En son fiyatları değerlendir
    recent = price_history[-window*3:]
    
    for i in range(window, len(recent)-window):
        # Lokal minimum (destek)
        if all(recent[i] <= recent[i-j] for j in range(1, window)) and all(recent[i] <= recent[i+j] for j in range(1, window)):
            supports.append(recent[i])
            
        # Lokal maksimum (direnç)
        if all(recent[i] >= recent[i-j] for j in range(1, window)) and all(recent[i] >= recent[i+j] for j in range(1, window)):
            resistances.append(recent[i])
    
    # En yakın 3 seviyeyi al
    current_price = price_history[-1]
    supports = [float(s) for s in supports[:3]]
    resistances = [float(r) for r in resistances[:3]]
    return supports, resistances

def calculate_bollinger_bands(price_history, window=20, num_std=2):
    """
    Bollinger Bantları hesaplama.
    """
    if len(price_history) < window:
        return None, None, None
    
    prices = np.array(price_history[-window*2:])
    sma = np.mean(prices[-window:])
    std = np.std(prices[-window:])
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return sma, upper_band, lower_band

def calculate_stochastic_oscillator(price_history, high_history=None, low_history=None, k_period=14, d_period=3):
    """
    Stokastik Osilatör hesaplama.
    """
    if len(price_history) < k_period:
        return 50, 50  # Varsayılan orta değerler
        
    # Use the last k_period values
    recent_prices = price_history[-k_period:]
    
    if high_history is None or low_history is None:
        # If high/low histories aren't provided, use price_history
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
    else:
        # Make sure we're getting the last k_period values
        try:
            recent_lows = low_history[-k_period:] if isinstance(low_history, list) else low_history
            recent_highs = high_history[-k_period:] if isinstance(high_history, list) else high_history
            
            # Extract scalar values, not lists
            if isinstance(recent_lows, list):
                if all(isinstance(x, (int, float)) for x in recent_lows):
                    lowest_low = min(recent_lows)
                else:
                    # If it's still a list of lists or other complex structure
                    lowest_low = min(recent_prices)
            else:
                lowest_low = recent_lows
                
            if isinstance(recent_highs, list):
                if all(isinstance(x, (int, float)) for x in recent_highs):
                    highest_high = max(recent_highs)
                else:
                    # If it's still a list of lists or other complex structure
                    highest_high = max(recent_prices)
            else:
                highest_high = recent_highs
        except Exception as e:
            # Fallback to price history in case of any error
            print(f"Error in stochastic calculation: {str(e)}")
            lowest_low = min(recent_prices)
            highest_high = max(recent_prices)
    
    close = price_history[-1]
    
    # Make sure we're working with scalar values
    if not isinstance(lowest_low, (int, float)) or not isinstance(highest_high, (int, float)):
        return 50, 50
    
    if highest_high == lowest_low:
        k_percent = 50
    else:
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate D% (simple moving average of K%)
    if len(price_history) >= k_period + d_period:
        k_values = []
        try:
            for i in range(d_period):
                day_idx = -1 - i
                if day_idx < -len(price_history):
                    # Skip if we're trying to access before the beginning of the array
                    continue
                
                day_close = price_history[day_idx]
                
                # Check if we have enough data for the window
                start_idx = day_idx - k_period + 1
                if start_idx < -len(price_history):
                    start_idx = -len(price_history)
                
                day_window = price_history[start_idx:day_idx+1]
                
                # Make sure we have data in the window
                if not day_window:
                    continue
                    
                # Protection against empty list
                if len(day_window) == 0:
                    continue
                    
                day_low = min(day_window)
                day_high = max(day_window)
                
                if day_high == day_low:
                    day_k = 50
                else:
                    day_k = 100 * ((day_close - day_low) / (day_high - day_low))
                
                k_values.append(day_k)
            
            # Only calculate if we have values
            d_percent = sum(k_values) / len(k_values) if k_values else k_percent
        except Exception as e:
            print(f"Error calculating stochastic D: {str(e)}")
            d_percent = k_percent
    else:
        d_percent = k_percent
    
    return float(k_percent), float(d_percent)

def identify_market_condition(price_history):
    """
    Piyasa durumunu belirleyen fonksiyon.
    """
    if len(price_history) < 50:
        return "TANIMSIZ"
    
    # Kısa ve uzun dönem eğilimler
    short_term = price_history[-10:] 
    medium_term = price_history[-30:]
    long_term = price_history[-50:]
    
    short_trend = (short_term[-1] / short_term[0] - 1) * 100
    medium_trend = (medium_term[-1] / medium_term[0] - 1) * 100
    long_trend = (long_term[-1] / long_term[0] - 1) * 100
    
    # Volatilite hesaplama
    volatility = np.std(price_history[-30:]) / np.mean(price_history[-30:]) * 100
    
    # Temel trendleri belirle
    if short_trend > 5 and medium_trend > 3:
        if volatility > 4:
            return "GÜÇLÜ YÜKSELİŞ - VOLATİL"
        return "GÜÇLÜ YÜKSELİŞ"
    elif short_trend < -5 and medium_trend < -3:
        if volatility > 4:
            return "SERT DÜŞÜŞ - VOLATİL"
        return "DÜŞÜŞ TRENDİ"
    elif abs(short_trend) < 2 and abs(medium_trend) < 3:
        if volatility < 2:
            return "YATAY - DÜŞÜK VOLATİLİTE"
        return "YATAY SEYİR"
    elif short_trend > 0 and medium_trend < 0:
        return "DİPTEN DÖNÜŞ"
    elif short_trend < 0 and medium_trend > 0:
        return "TEPE OLUŞUMU"
    else:
        return "KARARSIZ"

def analyze_opportunity(price_history, tech_score, news_score):
    """
    Alım ve satım fırsatlarını analiz eden gelişmiş fonksiyon.
    """
    if len(price_history) < 50:
        return {
            "signal": "VERİ YETERSİZ",
            "strength": 0,
            "reason": "Yeterli geçmiş veri yok"
        }
    
    # Temel göstergeler
    market_condition = identify_market_condition(price_history)
    current_price = price_history[-1]
    supports, resistances = detect_support_resistance(price_history)
    sma, upper_band, lower_band = calculate_bollinger_bands(price_history)
    stoch_k, stoch_d = calculate_stochastic_oscillator(price_history)
    
    # RSI hesaplama
    rsi = calculate_rsi(price_history)
    
    # Toplam skor ve sebep
    signals = []
    signal_strength = 0
    reasons = []
    
    # RSI tabanlı sinyaller
    if rsi < 30:
        signals.append("AL")
        signal_strength += 1
        reasons.append("Aşırı satım (RSI: {:.1f})".format(rsi))
    elif rsi > 70:
        signals.append("SAT")
        signal_strength -= 1
        reasons.append("Aşırı alım (RSI: {:.1f})".format(rsi))
        
    # Bollinger bantları tabanlı sinyaller
    if upper_band and lower_band:
        if current_price > upper_band:
            signals.append("SAT")
            signal_strength -= 1.5
            reasons.append("Üst Bollinger bandı aşıldı")
        elif current_price < lower_band:
            signals.append("AL")
            signal_strength += 1.5
            reasons.append("Alt Bollinger bandı aşıldı")
            
    # Stokastik osilatör tabanlı sinyaller
    if stoch_k < 20 and stoch_d < 20:
        signals.append("AL")
        signal_strength += 1
        reasons.append("Stokastik osilatör aşırı satım bölgesinde")
    elif stoch_k > 80 and stoch_d > 80:
        signals.append("SAT")
        signal_strength -= 1
        reasons.append("Stokastik osilatör aşırı alım bölgesinde")
        
    # Destek/direnç tabanlı sinyaller
    if supports and current_price < supports[0] * 1.03:
        signals.append("AL")
        signal_strength += 1
        reasons.append("Destek seviyesine yakın")
    if resistances and current_price > resistances[0] * 0.97:
        signals.append("SAT")
        signal_strength -= 1
        reasons.append("Direnç seviyesine yakın")
        
    # Piyasa koşulu ve skorlar
    if tech_score > 3 and news_score > 0:
        signals.append("AL")
        signal_strength += 1
        reasons.append("Teknik ve haber skorları pozitif")
    elif tech_score < -3 and news_score < 0:
        signals.append("SAT")
        signal_strength -= 1
        reasons.append("Teknik ve haber skorları negatif")
        
    # Toplam sinyali belirle
    final_signal = ""
    if signal_strength >= 3:
        final_signal = "GÜÇLÜ ALIM FIRSATI"
    elif signal_strength >= 1.5:
        final_signal = "ALIM FIRSATI"
    elif signal_strength <= -3:
        final_signal = "GÜÇLÜ SATIŞ SİNYALİ"
    elif signal_strength <= -1.5:
        final_signal = "SATIŞ SİNYALİ"
    elif abs(signal_strength) < 1.5:
        if "DÜŞÜŞ" in market_condition:
            final_signal = "İZLE (Düşüş eğilimi)"
        elif "YÜKSELİŞ" in market_condition:
            final_signal = "TÜTA (Yükseliş eğilimi)"
        else:
            final_signal = "NÖTR"
    
    # Hedef fiyatlar hesapla
    next_support = supports[0] if supports else current_price * 0.95
    next_resistance = resistances[0] if resistances else current_price * 1.05
    
    return {
        "signal": final_signal,
        "strength": float(signal_strength),  # numpy float'ı Python float'a çevir
        "market_condition": market_condition,
        "rsi": float(rsi),  # numpy.float64 -> float
        "stochastic": {
            "k": float(stoch_k), 
            "d": float(stoch_d)
        },
        "bollinger": {
            "sma": float(sma) if sma is not None else None,
            "upper": float(upper_band) if upper_band is not None else None,
            "lower": float(lower_band) if lower_band is not None else None
        },
        "targets": {
            "support": float(next_support),
            "resistance": float(next_resistance),
            "stop_loss": float(next_support * 0.98),
            "take_profit": float(next_resistance * 1.02)
        },
        "reasons": reasons
    }

if __name__ == '__main__':
    app.run(debug=False)
