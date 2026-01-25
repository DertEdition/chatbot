# Medical Chatbot - Docker Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler
- Docker Desktop yüklü olmalı
- GROQ API anahtarınız hazır olmalı

### 2. Kurulum

1. `.env` dosyası oluşturun:
```bash
# .env.example dosyasını kopyalayın
cp .env.example .env
```

2. `.env` dosyasını düzenleyin ve GROQ API anahtarınızı ekleyin:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Docker container'ı başlatın:
```bash
docker-compose up -d
```

### 3. Kullanım

Backend API şu adreste çalışacak:
- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### 4. Komutlar

```bash
# Container'ı başlat (detached mode)
docker-compose up -d

# Logları izle
docker-compose logs -f

# Container'ı durdur
docker-compose down

# Container'ı yeniden başlat
docker-compose restart

# Container'ı sil ve yeniden oluştur
docker-compose down
docker-compose up -d --build

# Container içine gir
docker-compose exec backend bash
```

### 5. Development Modu

`docker-compose.yml` dosyasında volumes tanımlı olduğu için, `app/` ve `data/` klasörlerindeki değişiklikler otomatik olarak container içine yansır. 

Uvicorn'u `--reload` modunda çalıştırmak için Dockerfile'daki CMD satırını şu şekilde değiştirin:
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 6. Production Modu

Production için:
1. `docker-compose.yml` dosyasındaki volumes satırlarını kaldırın veya yorum satırı yapın
2. `--reload` parametresini kaldırın
3. CORS ayarlarını `app/main.py` dosyasında güvenli hale getirin

## 🔧 Sorun Giderme

**Container başlamıyor:**
```bash
# Logları kontrol edin
docker-compose logs backend

# Container durumunu kontrol edin
docker ps -a
```

**Port zaten kullanımda:**
```bash
# docker-compose.yml dosyasında port'u değiştirin
ports:
  - "8001:8000"  # 8000 yerine 8001 kullan
```

**API anahtarı hatası:**
- `.env` dosyasının `chatbot/` klasöründe olduğundan emin olun
- GROQ_API_KEY değerinin doğru olduğunu kontrol edin

## 📝 Notlar

- Container ilk başlatmada model ve embedding dosyalarını indirecektir, bu biraz zaman alabilir
- Sentence-transformers ilk çalıştırmada model dosyalarını (~500MB) indirir
- Health check 40 saniye sonra başlar (model indirme süresi için)
