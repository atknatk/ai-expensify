# 💰 Cost Optimization Guide

Bu rehber, Invoice Analyzer API'sinin maliyetlerini nasıl düşürebileceğinizi gösterir.

## 📊 Maliyet Karşılaştırması (1M token başına)

| Provider | Model | Input | Output | Hız | Kalite | Not |
|----------|-------|-------|--------|-----|--------|-----|
| **Ollama** | llama3.1:8b | $0.00 | $0.00 | Orta | İyi | Local, GPU gerekli |
| **Groq** | llama3-8b-8192 | $0.05 | $0.08 | Çok Hızlı | İyi | En ucuz cloud |
| **Groq** | llama3-70b-8192 | $0.59 | $0.79 | Hızlı | Çok İyi | Kalite/fiyat dengesi |
| **Anthropic** | claude-3-haiku | $0.25 | $1.25 | Orta | Çok İyi | İyi kalite |
| **OpenAI** | gpt-4o-mini | $0.15 | $0.60 | Orta | İyi | Vision destekli |
| **OpenAI** | gpt-3.5-turbo | $0.50 | $1.50 | Hızlı | İyi | Standart |
| **OpenAI** | gpt-4o | $5.00 | $15.00 | Orta | Mükemmel | Vision premium |
| **OpenAI** | gpt-4 | $30.00 | $60.00 | Yavaş | Mükemmel | En pahalı |

## 🎯 Optimizasyon Stratejileri

### 1. **Ucuz Modeller Kullan**

`.env` dosyasında:
```bash
USE_CHEAP_MODELS=true
OCR_MODEL=gpt-4o-mini
TEXT_MODEL=gpt-3.5-turbo
VISION_MODEL=gpt-4o-mini
```

**Maliyet Tasarrufu:** ~80% (GPT-4'e göre)

### 2. **Alternatif LLM Servisleri**

#### A) Groq API (Çok Hızlı + Ucuz)
```bash
USE_ALTERNATIVE_LLMS=true
GROQ_API_KEY=your_groq_key
```

**Avantajlar:**
- 10x daha hızlı
- 10x daha ucuz
- İyi kalite

**Kurulum:**
1. https://console.groq.com/ adresine git
2. API key al (ücretsiz $25 kredit)
3. `.env` dosyasına ekle

#### B) Anthropic Claude (Kaliteli + Ucuz)
```bash
ANTHROPIC_API_KEY=your_anthropic_key
```

**Avantajlar:**
- GPT-4 kalitesinde
- 5x daha ucuz
- Türkçe desteği iyi

#### C) Ollama (Tamamen Ücretsiz!)
```bash
OLLAMA_BASE_URL=http://localhost:11434
```

**Kurulum:**
```bash
# Ollama'yı yükle
curl -fsSL https://ollama.ai/install.sh | sh

# Model indir
ollama pull llama3.1:8b

# Servis başlat
ollama serve
```

**Avantajlar:**
- Tamamen ücretsiz
- Veri gizliliği (local)
- İnternet bağımlılığı yok

**Dezavantajlar:**
- GPU gerekli (performans için)
- İlk kurulum karmaşık

### 3. **Akıllı Model Seçimi**

Sistem otomatik olarak görev tipine göre en uygun modeli seçer:

- **OCR/Vision:** gpt-4o-mini (vision gerekli)
- **Validation:** llama3.1:8b (basit görev)
- **Categorization:** llama3-8b-8192 (hızlı)
- **Summary:** llama3-70b-8192 (kaliteli)

### 4. **Token Optimizasyonu**

#### Prompt Kısaltma:
```python
# Uzun prompt (pahalı)
"Sen bir uzman fatura analiz sistemisin. Lütfen aşağıdaki faturayı analiz et..."

# Kısa prompt (ucuz)
"Faturayı analiz et:"
```

#### Max Token Limiti:
```python
max_tokens=1500  # Yerine 500-800 kullan
```

#### Temperature Düşürme:
```python
temperature=0.1  # Daha deterministik, daha az token
```

## 🚀 Hızlı Kurulum

### 1. Groq ile Başla (Önerilen)
```bash
# .env dosyasına ekle
USE_ALTERNATIVE_LLMS=true
GROQ_API_KEY=gsk_your_key_here

# API'yi yeniden başlat
poetry run uvicorn app.main:app --reload
```

### 2. Ollama ile Tamamen Ücretsiz
```bash
# Ollama kurulumu
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve

# .env dosyasına ekle
USE_ALTERNATIVE_LLMS=true
OLLAMA_BASE_URL=http://localhost:11434
```

## 📈 Maliyet Hesaplama

### Örnek Fatura Analizi:
- **OCR:** ~2000 token
- **Categorization:** ~500 token  
- **Validation:** ~800 token
- **Summary:** ~1500 token
- **Toplam:** ~4800 token

### Maliyet Karşılaştırması:
| Senaryo | Maliyet/Fatura | 1000 Fatura/Ay |
|---------|----------------|-----------------|
| GPT-4 (Pahalı) | $0.288 | $288 |
| GPT-3.5 (Standart) | $0.0096 | $9.6 |
| Groq (Ucuz) | $0.00048 | $0.48 |
| Ollama (Ücretsiz) | $0.00 | $0.00 |

**Tasarruf:** %99.8'e kadar!

## ⚡ Performans vs Maliyet

### Hız Sıralaması:
1. **Groq:** ~0.5 saniye
2. **GPT-3.5:** ~2 saniye  
3. **GPT-4o-mini:** ~3 saniye
4. **Ollama:** ~5 saniye (GPU'ya bağlı)
5. **GPT-4:** ~8 saniye

### Kalite Sıralaması:
1. **GPT-4:** Mükemmel
2. **Claude-3-Sonnet:** Çok İyi
3. **GPT-4o:** Çok İyi
4. **Llama3-70b:** İyi
5. **GPT-3.5:** İyi
6. **Llama3-8b:** Orta

## 🔧 Önerilen Konfigürasyon

### Düşük Maliyet (Önerilen):
```bash
USE_CHEAP_MODELS=true
USE_ALTERNATIVE_LLMS=true
GROQ_API_KEY=your_key
TEXT_MODEL=gpt-3.5-turbo
VISION_MODEL=gpt-4o-mini
```

### Sıfır Maliyet:
```bash
USE_ALTERNATIVE_LLMS=true
OLLAMA_BASE_URL=http://localhost:11434
# + Ollama kurulumu gerekli
```

### Yüksek Kalite (Pahalı):
```bash
USE_CHEAP_MODELS=false
TEXT_MODEL=gpt-4
VISION_MODEL=gpt-4o
```

## 📞 Destek

Maliyet optimizasyonu konusunda sorularınız için:
- GitHub Issues
- Documentation
- Community Discord

**Hedef:** %90+ maliyet tasarrufu ile aynı kalitede sonuçlar! 🎯
