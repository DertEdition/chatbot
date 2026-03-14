"""
Medical Chatbot - FastAPI Backend

Health-focused chatbot API using Groq LLM with translation pipeline.
Pipeline: TR → EN → LLM → EN → TR

This module provides the main FastAPI application for the medical chatbot,
including chat endpoints, translation services, and medicine name handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from groq import Groq
from deep_translator import GoogleTranslator

from app.health_filter import is_health_related, check_emergency_symptoms, is_non_health_topic, is_greeting, get_greeting_type, count_health_signals, count_non_health_signals
from app.prompts import get_system_prompt, format_response_prompt, get_greeting_response
from app.medicines import MEDICINE_BRANDS
from app.medicine_utils import detect_medicines, mask_medicines, unmask_medicines, convert_english_medicines_to_turkish
from app.domain import check_health_domain_simple

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Local model configuration
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

# Colab fine-tuned model configuration
USE_COLAB_MODEL = os.getenv("USE_COLAB_MODEL", "false").lower() == "true"
COLAB_API_URL = os.getenv("COLAB_API_URL", "")

if USE_COLAB_MODEL and COLAB_API_URL:
    print(f"🧪 COLAB MODEL MODE - Fine-tuned Llama 3.1 8B")
    print(f"   URL: {COLAB_API_URL}")
    groq_client = None
elif USE_LOCAL_MODEL:
    print("🏠 LOCAL MODEL MODE - Fine-tuned Qwen2.5-3B")
    groq_client = None
else:
    if not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not set! Add it to .env file.")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"☁️  GROQ API MODE - Model: {GROQ_MODEL}")

# Translation clients for Turkish-English conversion
tr_to_en = GoogleTranslator(source='tr', target='en')
en_to_tr = GoogleTranslator(source='en', target='tr')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    print("🚀 Medical Chatbot API starting...")
    yield  # Application runs here
    print("👋 Shutting down...")


app = FastAPI(
    title="Medical Chatbot API",
    description="Health-focused informational chatbot - Groq + Translation + RAG",
    version="3.0.0",
    lifespan=lifespan
)

# Include RAG router (optional - only if RAG dependencies are installed)
try:
    from app.rag.router import router as rag_router
    app.include_router(rag_router)
    print("✅ RAG router loaded - /rag/* endpoints active")
except ImportError as e:
    print(f"⚠️ RAG router not loaded (sentence-transformers/faiss not installed): {e}")

# Include Vision router (drug image analysis)
try:
    from app.vision_router import router as vision_router
    app.include_router(vision_router)
    print("✅ Vision router loaded - /vision/* endpoints active")
except ImportError as e:
    print(f"⚠️ Vision router not loaded: {e}")

# CORS configuration
# NOTE: In production, change allow_origins to a whitelist
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    """Represents a single message in the chat conversation."""
    role: str  # "user" or "assistant"
    content: str
    content_en: Optional[str] = None  # English version for drift prevention


class SymptomContext(BaseModel):
    """
    Structured symptom information from the 3D body model.

    Contains detailed information about the user's symptom selection
    from the interactive 3D human body interface.
    """
    region: str  # e.g., "left_shin"
    region_name_tr: str  # e.g., "Sol Kaval Kemiği"
    region_name_en: str  # e.g., "Left Shin (Tibia)"
    symptom: str  # e.g., "pain"
    symptom_name_tr: str  # e.g., "Ağrı"
    symptom_name_en: str  # e.g., "Pain"
    severity_0_10: int  # Severity scale from 0 to 10
    onset: str  # e.g., "2_3_days"
    trigger: Optional[str] = None  # e.g., "after_running"
    red_flags: List[str] = Field(default_factory=list)  # List of reported red flags


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.

    Attributes:
        message: The user's message text
        history: Previous conversation messages for context
        detailed_response: Whether to return a detailed response format
        symptom_context: Structured symptom data from 3D body model
    """
    message: str
    history: List[Message] = Field(default_factory=list)
    detailed_response: bool = False
    symptom_context: Optional[SymptomContext] = None


class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.

    Attributes:
        response: The Turkish response text
        response_en: English version for drift prevention (stored by frontend)
        is_emergency: Whether an emergency was detected
        disclaimer: Medical disclaimer text
        model_provider: Which model generated the response
    """
    response: str
    response_en: Optional[str] = None
    is_emergency: bool = False
    disclaimer: str = "⚠️ Bu bilgiler eğitim amaçlıdır, tıbbi tavsiye değildir. Acil durumlarda 112'yi arayın."
    model_provider: Optional[str] = None

def translate_to_english(text: str) -> str:
    """
    Translate Turkish text to English.

    Medicine masks (MEDTOK tokens) are preserved during translation.

    Args:
        text: Turkish text to translate

    Returns:
        Translated English text, or original text if translation fails
    """
    try:
        translated = tr_to_en.translate(text)
        print(f"[TR→EN] {text[:50]}... → {translated[:50]}...")
        return translated
    except Exception as e:
        print(f"[ERROR] Translation error (TR→EN): {e}")
        return text


def translate_to_turkish(text: str) -> str:
    """
    Translate English text to Turkish.

    Args:
        text: English text to translate

    Returns:
        Translated Turkish text, or original text if translation fails
    """
    try:
        translated = en_to_tr.translate(text)
        print(f"[EN→TR] {text[:50]}... → {translated[:50]}...")
        return translated
    except Exception as e:
        print(f"[ERROR] Translation error (EN→TR): {e}")
        return text


def get_rag_context_for_colab(question: str) -> str:
    """
    RAG knowledge base'den fine-tuned model için bağlam al.
    RAG yüklü değilse veya hata olursa boş string döner.
    """
    try:
        from app.rag.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        if kb is None:
            return ""

        results = kb.search(query=question, top_k=3)
        if not results:
            return ""

        # En iyi 3 sonucu context olarak formatla
        context_parts = []
        for i, doc in enumerate(results[:3], 1):
            title = doc.get("metadata", {}).get("title", "")
            text = doc.get("text", "")
            if text:
                # Çok uzun textleri kısalt
                if len(text) > 500:
                    text = text[:500] + "..."
                context_parts.append(f"[Source {i}: {title}]\n{text}")

        context = "\n---\n".join(context_parts)
        print(f"[RAG→COLAB] Found {len(results)} results, using top {len(context_parts)}")
        return context

    except Exception as e:
        print(f"[RAG→COLAB] RAG not available: {e}")
        return ""


def build_enriched_prompt(question: str, rag_context: str = "") -> str:
    """
    Prompt Engineering: 8B model için basit ve etkili prompt.
    Karmaşık talimatlar yerine RAG bilgisini doğrudan soruya ekle.
    """
    # RAG bilgisini soruya doğal şekilde entegre et
    if rag_context:
        return (
            f"Based on the following medical information:\n{rag_context}\n\n"
            f"Answer this question: {question}"
        )
    else:
        return question


def post_process_response(text: str) -> str:
    """
    Agresif post-processing: Fine-tuned model çıktısını temizle.
    Sıralama önemli: ÖNCE kes, SONRA temizle.
    """
    import re

    if not text:
        return text

    # === STEP 1: İlk disclaimer'da kes (EN ÖNEMLİ) ===
    # Model: iyi cevap → disclaimer → saçmalama. İlk disclaimer'da kesiyoruz.
    cut_patterns = [
        r'\*This is general health information[^*]*\*',
        r'\*Always seek immediate attention[^*]*\*',
        r'\*Consult a healthcare professional[^*]*\*',
        r'\*If symptoms persist[^*]*\*',
        r'\*Never delay seeking help[^*]*\*',
        r'\*Disclaimer:[^*]*\*',
        r'\*This information is for educational[^*]*\*',
    ]
    for pattern in cut_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.end()]
            break

    # === STEP 2: Stop phrase'lerde kes ===
    stop_phrases = [
        "Is there anything else I can help you with",
        "Do you have any other questions",
        "Feel free to ask",
        "I hope this helps",
        "Let me know if you",
        "If you have any further",
        "Please don't hesitate",
    ]
    for phrase in stop_phrases:
        idx = text.lower().find(phrase.lower())
        if idx > 0:
            end = idx + len(phrase)
            for i in range(end, min(end + 50, len(text))):
                if text[i] in '.?!\n':
                    end = i + 1
                    break
            text = text[:end]
            break

    # === STEP 3: Kalan disclaimer'ları kaldır (artık sadece tek disclaimer kaldı) ===
    disclaimer_patterns = [
        r'\*Disclaimer:.*?\*',
        r'\*This information is for educational.*?\*',
        r'\*This is general health information.*?\*',
        r'\*Please consult a healthcare professional.*?\*',
        r'\*Always consult.*?advice\.?\*',
        r'\*Always seek immediate attention.*?\*',
        r'\*Never delay seeking help.*?\*',
        r'\*If this is an emergency.*?\*',
        r'\*Consult your doctor.*?\*',
        r'\*This is not medical advice.*?\*',
        r'\*Do not rely on this.*?\*',
        r'\*If symptoms persist.*?\*',
        r'\*Please do not self-diagnose.*?\*',
        r'\*Note:.*?professional.*?\*',
        r'Disclaimer:.*?(?:\n|$)',
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # === STEP 4: Tekrarlayan satırları kaldır ===
    lines = text.split('\n')
    seen = set()
    deduped = []
    empty_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            empty_count += 1
            if empty_count <= 2:
                deduped.append(line)
            continue
        empty_count = 0
        normalized = re.sub(r'\s+', ' ', stripped.lower())
        if len(normalized) > 15 and normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(line)
    text = '\n'.join(deduped)

    # === STEP 5: Uzunluk limiti ===
    if len(text) > 2000:
        last_para = text[:2000].rfind('\n\n')
        if last_para > 500:
            text = text[:last_para]

    # === STEP 6: Final temizlik ===
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def call_colab_model(messages: list, system_prompt: str = None) -> str:
    """
    Send a request to the Colab-hosted fine-tuned model via Gradio 5.x API.
    Enhanced with: Prompt Engineering + RAG Context + Post-processing.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt

    Returns:
        The model response text

    Raises:
        HTTPException: If the API call fails
    """
    import requests as http_requests
    import json

    try:
        question = messages[-1]["content"] if messages else ""

        print(f"[COLAB] Sending question to {COLAB_API_URL}")

        is_modal = "modal.run" in COLAB_API_URL

        if is_modal:
            # Modal endpoint: simple JSON API
            r = http_requests.post(
                COLAB_API_URL,
                json={"message": question, "max_tokens": 512, "temperature": 0.7},
                timeout=300,
                allow_redirects=True,
            )

            if r.status_code != 200:
                raise HTTPException(status_code=503, detail=f"Modal model error: {r.status_code}")

            data = r.json()
            result = data.get("response", "")
        else:
            # Gradio endpoint (Colab): async SSE API
            r = http_requests.post(
                f"{COLAB_API_URL}/gradio_api/call/chat",
                json={"data": [question, [], "You are a helpful medical assistant. Provide clear, structured health information. For emergencies, advise calling 112. Recommend professional consultation when appropriate.", 512, 0.7]},
                timeout=30
            )

            if r.status_code != 200:
                raise HTTPException(status_code=503, detail="Colab model unavailable")

            event_id = r.json().get("event_id")

            r2 = http_requests.get(
                f"{COLAB_API_URL}/gradio_api/call/chat/{event_id}",
                timeout=300,
                stream=True
            )

            result = ""
            for line in r2.iter_lines():
                if line:
                    decoded = line.decode()
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data != "null":
                            parsed = json.loads(data)
                            if isinstance(parsed, list) and len(parsed) >= 2:
                                history = parsed[1]
                                if history and isinstance(history, list) and history[-1]:
                                    result = history[-1][1] if history[-1][1] else ""
                            elif isinstance(parsed, list) and parsed:
                                result = parsed[0] if parsed[0] else ""

        # === 3. Post-processing (temizle, düzenle) ===
        result = post_process_response(result)
        print(f"[COLAB] Response: {result[:100]}...")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Colab model error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Colab model error: {str(e)}")


def is_quality_response(text: str) -> bool:
    """
    Check if a model response has enough meaningful content.
    Returns False if the response is too short, empty, or mostly filler.
    """
    if not text or len(text.strip()) < 80:
        return False

    # Check if it's mostly generic "consult a doctor" with no real content
    generic_phrases = [
        "consult a healthcare", "see a doctor", "seek medical",
        "consult your doctor", "medical professional",
        "we cannot provide", "i cannot provide", "not a licensed",
        "general advice", "general health information",
    ]
    lower = text.lower()
    generic_count = sum(1 for p in generic_phrases if p in lower)
    # If more than half the sentences are generic disclaimers, it's bad
    sentence_count = max(1, lower.count('.') + lower.count('?') + lower.count('!'))
    if generic_count >= sentence_count * 0.5 and sentence_count < 5:
        return False

    return True


def _call_groq_api(messages: list, system_prompt: str = None) -> str:
    """Direct Groq API call (no routing logic)."""
    groq_messages = []
    if system_prompt:
        groq_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    print(f"[GROQ] Sending request, model: {GROQ_MODEL}")
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=groq_messages,
        temperature=0.7,
        max_tokens=2048,
    )
    result = response.choices[0].message.content
    print(f"[GROQ] Response: {result[:100]}...")
    return result


def call_groq(messages: list, system_prompt: str = None) -> tuple:
    """
    Send a request to the LLM (Groq API, Local Model, or Colab).

    Smart fallback: If Colab model produces a low-quality response,
    automatically falls back to Groq+RAG for reliable output.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend to messages

    Returns:
        Tuple of (response_text, provider_name)

    Raises:
        HTTPException: If the API call fails
    """
    # Use Colab fine-tuned model if configured (with Groq fallback)
    if USE_COLAB_MODEL and COLAB_API_URL:
        try:
            result = call_colab_model(messages, system_prompt)
            if is_quality_response(result):
                return result, "colab-finetuned (Llama-3.1-8B + LoRA)"
            else:
                print(f"[FALLBACK] Colab response too short/low quality ({len(result)} chars), falling back to Groq")
        except Exception as e:
            print(f"[FALLBACK] Colab model failed: {e}, falling back to Groq")

        # Fallback to Groq
        if groq_client is None:
            # Initialize Groq client for fallback (wasn't created at startup because Colab was primary)
            from groq import Groq as GroqClient
            fallback_client = GroqClient(api_key=GROQ_API_KEY)
            groq_messages = []
            if system_prompt:
                groq_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                groq_messages.append({"role": msg["role"], "content": msg["content"]})
            print(f"[FALLBACK] Using Groq API (fallback), model: {GROQ_MODEL}")
            response = fallback_client.chat.completions.create(
                model=GROQ_MODEL, messages=groq_messages, temperature=0.7, max_tokens=2048
            )
            result = response.choices[0].message.content
            return result, f"groq-fallback ({GROQ_MODEL})"
        else:
            result = _call_groq_api(messages, system_prompt)
            return result, f"groq-fallback ({GROQ_MODEL})"

    # Use local model if configured
    if USE_LOCAL_MODEL:
        result = call_local_model(messages, system_prompt)
        return result, "local-finetuned (Qwen2.5-3B + LoRA)"

    # Use Groq API (primary)
    try:
        result = _call_groq_api(messages, system_prompt)
        return result, f"groq ({GROQ_MODEL})"
    except Exception as e:
        print(f"[ERROR] Groq error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"LLM API error: {str(e)}")


def call_local_model(messages: list, system_prompt: str = None) -> str:
    """
    Send a request to the local fine-tuned model.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend to messages

    Returns:
        The LLM response text

    Raises:
        HTTPException: If model inference fails
    """
    try:
        from app.local_model import generate_response

        print(f"[DEBUG] Using local fine-tuned model")

        result = generate_response(
            messages=messages,
            system_prompt=system_prompt,
            max_new_tokens=1024,
            temperature=0.7,
        )

        print(f"[DEBUG] Local model response: {result[:100]}...")
        return result

    except Exception as e:
        print(f"[ERROR] Local model error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Local model error: {str(e)}")


def call_groq_classifier(messages: list, system_prompt: str) -> str:
    """
    Optimized Groq API call for classification tasks.

    Uses deterministic settings for consistent classification:
    - temperature=0 for deterministic output
    - max_tokens=3 for YES/NO/UNCERTAIN response
    - stop=["\\n"] for single-line response

    Args:
        messages: List of message dictionaries
        system_prompt: System prompt for classification

    Returns:
        Classification result: "YES", "NO", or "UNCERTAIN"
    """
    try:
        groq_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})

        print("[CLASSIFIER] Sending classification request to Groq")

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=groq_messages,
            temperature=0,
            max_tokens=3,
            stop=["\n"],
        )

        result = response.choices[0].message.content.strip().upper()
        print(f"[CLASSIFIER] Result: {result}")
        return result

    except Exception as e:
        print(f"[ERROR] Classifier error: {str(e)}")
        return "UNCERTAIN"





def get_english_system_prompt(detailed: bool = False, has_history: bool = False, symptom_context: SymptomContext = None) -> str:
    """
    Generate the English system prompt for the LLM.

    Returns different prompts for initial questions vs follow-up questions.
    If symptom_context is provided, includes structured data from the 3D body model.

    Args:
        detailed: Whether to request a detailed response format
        has_history: Whether this is a follow-up question (has conversation history)
        symptom_context: Optional structured symptom data from 3D model

    Returns:
        The system prompt string for the LLM
    """
    context_section = ""
    if symptom_context:
        context_section = f"""
=== STRUCTURED SYMPTOM DATA FROM 3D BODY MODEL ===
The user has selected the following through the interactive 3D human body interface:

BODY REGION: {symptom_context.region_name_en} ({symptom_context.region})
SYMPTOM TYPE: {symptom_context.symptom_name_en} ({symptom_context.symptom})
SEVERITY: {symptom_context.severity_0_10}/10
ONSET: {symptom_context.onset}
TRIGGER: {symptom_context.trigger or 'Not specified'}
RED FLAGS REPORTED: {', '.join(symptom_context.red_flags) if symptom_context.red_flags else 'None'}

Use this structured data to provide more accurate and targeted guidance.
Focus on the specific body region and symptom type.
If red flags are present, emphasize seeking immediate medical attention.
=================================================

"""
    
    if not has_history:
        # İLK SORU - Kapsamlı yanıt
        return context_section + """You are a medical health assistant. Your role is to provide health education and general guidance.

IMPORTANT: This is the user's FIRST question. Provide a COMPREHENSIVE response with this EXACT structure:

**Your concern:** [1-2 sentence acknowledgment and brief explanation]

**Possible Causes:**
• [Cause 1]
• [Cause 2]
• [Cause 3]
• [Cause 4]

**What You Can Do:**
• [Recommendation 1]
• [Recommendation 2]
• [Recommendation 3]
• [Recommendation 4]

**Questions for You:**
• [Question about duration]
• [Question about severity]
• [Question about other symptoms]

**⚠️ Warning Signs - See a Doctor If:**
• [Red flag 1]
• [Red flag 2]
• [Red flag 3]
• [Red flag 4]

FORMATTING RULES:
- ALWAYS use bullet points (•) for lists - NEVER write as paragraphs
- Use **bold** for section headers
- Keep each bullet point to 1-2 sentences max
- Be empathetic but concise
- Do NOT diagnose or prescribe
- You are NOT a doctor"""
    
    else:
        # TAKİP SORUSU - Odaklı yanıt
        return context_section + """You are a medical health assistant continuing a conversation.

IMPORTANT: This is a FOLLOW-UP question. Be CONCISE and FOCUSED.

**Response Format:**
- Start with a direct answer to their question
- Use bullet points when listing multiple items:
  • Point 1
  • Point 2
- Keep response to 3-5 bullet points or 2-3 short paragraphs
- Don't repeat information already given

**If they share new symptoms:**
• Acknowledge the new info briefly
• Adjust guidance if needed
• Mention if urgency changes

RULES:
- You are NOT a doctor
- Be concise - this is a follow-up, not a new consultation
- Use bullet points (•) for any lists
- Stay focused on their current question"""


@app.get("/")
async def root():
    return {"message": "Medical Chatbot API", "status": "active", "provider": "Groq + Translation"}


def has_health_context_in_history(history: list) -> bool:
    """
    Check if conversation history contains actual health-related topics.

    Returns False if history only contains greetings or casual messages.

    Args:
        history: List of Message objects from conversation history

    Returns:
        True if history contains health-related content, False otherwise
    """
    if not history:
        return False

    for msg in history:
        if msg.role == "user":
            content = msg.content.lower()
            if not is_greeting(content) and is_health_related(content):
                return True

    return False


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the medical chatbot.

    Pipeline: TR Question → EN Translation → Groq LLM → TR Translation → Response

    The endpoint handles:
    - Greeting detection and responses
    - Emergency symptom detection
    - Health domain validation
    - Medicine name masking/unmasking for accurate translation
    - Context-aware responses based on conversation history

    Args:
        request: ChatRequest containing message, history, and optional symptom context

    Returns:
        ChatResponse with Turkish response and optional English version

    Raises:
        HTTPException: If message is empty or API call fails
    """
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    has_history = len(request.history) > 0
    has_health_context = has_health_context_in_history(request.history)
    has_symptom_context = request.symptom_context is not None

    # Step 1: Greeting check (Turkish)
    # Only respond with greeting if no symptom context and no health context
    greeting_type = get_greeting_type(user_message)
    if greeting_type and not has_health_context and not has_symptom_context:
        return ChatResponse(
            response=get_greeting_response(greeting_type),
            is_emergency=False
        )

    # Step 2: Emergency check (Turkish + Structured context)
    # Check red flags from structured context
    if request.symptom_context and request.symptom_context.red_flags:
        critical_flags = ['loss_of_consciousness', 'difficulty_breathing', 'chest_pain', 'severe_bleeding']
        if any(flag in critical_flags for flag in request.symptom_context.red_flags):
            return ChatResponse(
                response=f"🚨 **ACİL DURUM UYARISI** 🚨\n\nBildirdiğiniz belirtiler ({request.symptom_context.region_name_tr} - {request.symptom_context.symptom_name_tr}) acil tıbbi müdahale gerektirebilir!\n\n**HEMEN 112'yi arayın veya en yakın acil servise gidin!**\n\n⚠️ Bu durumu ciddiye alın ve beklemeden profesyonel yardım alın.",
                is_emergency=True,
                disclaimer="🚨 ACİL DURUM - Hemen 112'yi arayın!"
            )
    
    is_emergency, emergency_response = check_emergency_symptoms(user_message)
    if is_emergency:
        return ChatResponse(
            response=emergency_response,
            is_emergency=True,
            disclaimer="🚨 ACİL DURUM - Hemen 112'yi arayın!"
        )

    # Step 3: Health domain validation
    # If symptom_context exists, automatically accept as health topic
    # - Initial health question: perform full health check
    # - Follow-ups: only reject clearly unrelated topics 
    #   Short answers like "yes", "3 days" are accepted
    has_symptom_context = request.symptom_context is not None

    if not is_greeting(user_message) and not has_symptom_context:
        if has_health_context:
            # Follow-up: only reject clear non-health topic changes
            # First check if there's any health signal
            health_kw, health_pat, _, _ = count_health_signals(user_message)
            hard_nh, soft_nh, _, _ = count_non_health_signals(user_message)

            # If health signal exists, continue
            if health_kw + health_pat > 0:
                pass
            # In follow-up, only reject HARD topic changes
            elif hard_nh > 0:
                return ChatResponse(
                    response="Anladım, konu değiştirmek istiyorsunuz. 😊\n\nAncak ben sadece sağlık konularında yardımcı olabiliyorum. Eğer sağlıkla ilgili başka bir sorunuz varsa, sormaktan çekinmeyin!\n\nÖnceki konuya devam etmek isterseniz de yanınızdayım.",
                    is_emergency=False
                )
            # Soft non-health signals - don't reject in follow-up
            else:
                pass
        else:
            # Initial health question (or only greeting history): full health check
            domain_result = check_health_domain_simple(user_message)
            
            if domain_result == "NO":
                return ChatResponse(
                    response="Merhaba! Ben sağlık odaklı bir asistanım. 🏥\n\nSadece sağlık, hastalık, semptom ve tedavi ile ilgili sorularınızda size yardımcı olabilirim. Sağlık dışı konularda maalesef yardımcı olamıyorum.\n\nSağlıkla ilgili bir sorunuz varsa, lütfen sorun!",
                    is_emergency=False
                )
            elif domain_result == "UNCERTAIN":
                return ChatResponse(
                    response="Merhaba! 😊 Mesajınızı tam anlayamadım.\n\nBen sağlık konularında yardımcı olan bir asistanım. Sağlık, semptom veya ilaçlarla ilgili bir sorunuz mu var?\n\nLütfen sorunuzu biraz daha açıklayabilir misiniz?",
                    is_emergency=False
                )

    # Step 4: Translation Pipeline
    # TR → MASK → EN → LLM → TR → UNMASK → EN→TR
    # Mask medicine names, translate, get LLM response, translate back, unmask

    # Global mask map and counter (single map for history + current message)
    global_mask_map = {}
    mask_counter = 0

    # Step 4a: Process history messages (start from history, prevent counter collision)
    messages_en = []
    for msg in request.history[-10:]:
        if msg.content_en:
            # Use English version from frontend (drift prevention)
            content_en = msg.content_en
        elif msg.role == "user":
            # User message: mask and translate (continue counter)
            masked_hist, global_mask_map, mask_counter = mask_medicines(
                msg.content, start_counter=mask_counter, existing_mask_map=global_mask_map
            )
            content_en = translate_to_english(masked_hist)
        else:
            # Assistant message without content_en: translate (backward compatibility)
            content_en = translate_to_english(msg.content)

        messages_en.append({"role": msg.role, "content": content_en})

    # Step 4b: Mask medicines in user message (continue counter from history)
    masked_message, global_mask_map, mask_counter = mask_medicines(
        user_message, start_counter=mask_counter, existing_mask_map=global_mask_map
    )
    print(f"[MASK-MAP] {global_mask_map}")

    # Step 4c: Translate masked message to English
    user_message_en = translate_to_english(masked_message)

    messages_en.append({"role": "user", "content": user_message_en})

    # Step 4d: Get English system prompt (with structural context)
    # has_health_context: True = follow-up (concise), False = initial question (detailed)
    system_prompt_en = get_english_system_prompt(
        detailed=request.detailed_response,
        has_history=has_health_context,
        symptom_context=request.symptom_context
    )

    # Step 4e: Get English response from LLM
    response_en_raw, model_provider = call_groq(messages_en, system_prompt=system_prompt_en)

    # Step 4f: Translate response to Turkish
    response_tr = translate_to_turkish(response_en_raw)

    # Step 4g: Convert LLM-generated English medicine names to Turkish
    # (names not caught by mask like "ibuprofen", "acetaminophen")
    # NOTE: This must happen BEFORE unmask to avoid double conversion
    response_tr = convert_english_medicines_to_turkish(response_tr, format_style="tr_with_en")

    # Step 4h: Unmask medicine tokens: MEDTOK0 → "Parol (paracetamol)"
    if global_mask_map:
        response_tr = unmask_medicines(response_tr, global_mask_map, format_style="tr_with_en")
        # For response_en, use en_only (drift prevention - keep pure English)
        response_en_raw = unmask_medicines(response_en_raw, global_mask_map, format_style="en_only")

    return ChatResponse(
        response=response_tr,
        response_en=response_en_raw,
        is_emergency=False,
        model_provider=model_provider
    )


@app.get("/models")
async def list_models():
    """
    List available LLM models and current configuration.

    Returns:
        Dictionary with current model, available models, provider, and pipeline info
    """
    if USE_COLAB_MODEL and COLAB_API_URL:
        return {
            "current_model": "Llama-3.1-8B-Instruct + LoRA",
            "mode": "colab",
            "adapter": "bilgehantekin/medical-llama-3.1-8b-adapter",
            "provider": "Google Colab (Fine-tuned)",
            "pipeline": "TR → EN → LLM → TR",
            "colab_url": COLAB_API_URL
        }
    elif USE_LOCAL_MODEL:
        return {
            "current_model": "Qwen2.5-3B-Instruct + LoRA",
            "mode": "local",
            "adapter": "checkpoint-1000 (fine-tuned)",
            "provider": "Local (Fine-tuned)",
            "pipeline": "TR → EN → LLM → TR"
        }
    else:
        return {
            "current_model": GROQ_MODEL,
            "mode": "cloud",
            "available_models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768"
            ],
            "provider": "Groq",
            "pipeline": "TR → EN → LLM → TR"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
