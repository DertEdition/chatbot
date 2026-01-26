# Medical Chatbot Backend

Sağlık odaklı yapay zeka chatbot API'si. FastAPI üzerine kurulu, Groq LLM ve RAG (Retrieval-Augmented Generation) sistemi ile desteklenmektedir.

## Özellikler

- **Çoklu Dil Desteği**: Türkçe-İngilizce çeviri pipeline'ı (TR → EN → LLM → TR)
- **RAG Sistemi**: OpenFDA ve MedlinePlus verilerinden zenginleştirilmiş bilgi tabanı
- **Akıllı İlaç İşleme**: Türkçe ilaç isimlerini koruyarak doğru çeviri
- **Acil Durum Algılama**: Kritik semptomları otomatik tespit
- **3D Vücut Modeli Entegrasyonu**: Yapılandırılmış semptom verisi desteği
- **Sağlık Domain Filtresi**: Sadece sağlık konularında yanıt

## Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Framework | FastAPI |
| LLM | Groq (Llama 3.3 70B) |
| Embedding | Sentence Transformers |
| Vector Store | FAISS |
| Çeviri | Google Translator |

## Kurulum

### 1. Gereksinimler

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri

`.env.example` dosyasını `.env` olarak kopyalayın ve API anahtarlarınızı ekleyin:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Groq API anahtarı almak için: https://console.groq.com

### 3. Çalıştırma

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Ana Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | API durumu |
| GET | `/health` | Health check |
| POST | `/chat` | Ana sohbet endpoint'i |
| GET | `/models` | Kullanılabilir modeller |

### RAG Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/rag/chat` | RAG destekli sohbet |
| GET | `/rag/stats` | RAG istatistikleri |

## Kullanım Örneği

### Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Baş ağrısı için ne yapmalıyım?",
    "history": []
  }'
```

### 3D Model ile Yapılandırılmış Veri

```json
{
  "message": "Bu ağrı ne anlama geliyor?",
  "symptom_context": {
    "region": "head",
    "region_name_tr": "Baş",
    "region_name_en": "Head",
    "symptom": "pain",
    "symptom_name_tr": "Ağrı",
    "symptom_name_en": "Pain",
    "severity_0_10": 7,
    "onset": "2_3_days",
    "trigger": "stress",
    "red_flags": []
  }
}
```

## Proje Yapısı

```
backend/
├── app/
│   ├── main.py              # FastAPI uygulaması
│   ├── health_filter.py     # Sağlık domain filtresi
│   ├── prompts.py           # LLM prompt şablonları
│   ├── medicines.py         # İlaç veritabanı
│   ├── medicine_utils.py    # İlaç işleme araçları
│   ├── domain.py            # Domain sınıflandırma
│   └── rag/
│       ├── router.py        # RAG API endpoint'leri
│       ├── rag_chain.py     # RAG pipeline
│       ├── knowledge_base.py # Bilgi tabanı yönetimi
│       ├── vector_store.py  # FAISS vector store
│       ├── embeddings.py    # Embedding işlemleri
│       └── performance.py   # Performans izleme
├── data/
│   └── medical_knowledge/   # Tıbbi bilgi JSON dosyaları
├── scripts/
│   └── etl/                 # Veri işleme scriptleri
├── tests/                   # Test dosyaları
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Docker ile Çalıştırma

```bash
docker build -t medical-chatbot-backend .
docker run -p 8000:8000 --env-file .env medical-chatbot-backend
```

## Veri Kaynakları

- **OpenFDA**: İlaç bilgileri (Türkçe'ye çevrilmiş)
- **MedlinePlus**: Hastalık ve semptom bilgileri

## Lisans

Bu proje eğitim amaçlıdır. Tıbbi tavsiye niteliği taşımaz.

---

**Uyarı**: Bu chatbot tıbbi teşhis veya tedavi sağlamaz. Acil durumlarda 112'yi arayın.
