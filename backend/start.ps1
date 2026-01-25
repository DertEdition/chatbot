# Medical Chatbot - Hızlı Başlatma Script'i
Write-Host "🚀 Sunucu başlatılıyor..." -ForegroundColor Green
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
