# Python 3.11 imajını kullan
FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle (sentence-transformers için gerekli)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app/ ./app/
COPY data/ ./data/

# .env dosyası varsa kopyala (opsiyonel - docker-compose'da env vars kullanılabilir)
COPY .env* ./

# Port'u expose et
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
