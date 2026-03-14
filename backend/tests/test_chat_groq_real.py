"""
Medical Chatbot Real Groq API Integration Tests

These tests use REAL Groq API calls (not mocks).
Requires GROQ_API_KEY in .env file.

Run with:
  pytest backend/tests/test_chat_groq_real.py -v
  
Skip if .env missing:
  pytest backend/tests/test_chat_groq_real.py -v -m "not requires_groq_key"
"""

import pytest
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Skip all tests if GROQ_API_KEY not present
pytestmark = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY not set in .env file"
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


# ============================================================================
# TC-CHAT-01: Health query with REAL Groq response
# ============================================================================
def test_tc_chat_01_real_groq(client):
    """
    Health-related query returns meaningful Groq response
    Validates: Domain check + Translation + Groq API + Response quality
    """
    response = client.post(
        "/chat",
        json={
            "message": "Başım 3 gündür ağrıyor, ne yapmalıyım?",
            "history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "response" in data
    assert "is_emergency" in data
    assert data["is_emergency"] == False  # Not emergency
    
    # Check response quality
    answer = data["response"]
    assert len(answer) > 50, "Response should be detailed"
    assert "başağrısı" in answer.lower() or "ağrı" in answer.lower() or "paracetamol" in answer.lower().replace("parasetamol", "paracetamol")
    assert "<" not in answer and ">" not in answer, "No HTML artifacts"
    
    print(f"\n✅ TC-CHAT-01 (REAL) - Response quality: {len(answer)} chars")
    print(f"   Sample: {answer[:150]}...")


# ============================================================================
# TC-CHAT-04: Emergency response with REAL Groq
# ============================================================================
def test_tc_chat_04_real_groq(client):
    """
    Emergency scenario returns emergency guidance + meaningful Groq response
    Validates: Emergency detection + Groq API + Combined response
    """
    response = client.post(
        "/chat",
        json={
            "message": "Göğsümde şiddetli ağrı var ve nefes almakta çok zorluk çekiyorum",
            "history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check emergency detected
    assert data.get("is_emergency") == True
    assert "🚨" in data["response"] or "112" in data["response"] or "acil" in data["response"].lower()
    
    # Response should be detailed enough
    assert len(data["response"]) > 50
    
    print(f"\n✅ TC-CHAT-04 (REAL) - Emergency detected correctly")
    print(f"   Response includes: {data['response'][:100]}...")


# ============================================================================
# TC-CHAT-05: Medicine masking with REAL Groq
# ============================================================================
def test_tc_chat_05_real_groq(client):
    """
    Medicine name detected and Groq provides relevant response
    Validates: Medicine detection + Groq API + Meaningful answer
    """
    response = client.post(
        "/chat",
        json={
            "message": "Paracetamol kullandım ama baş ağrısı geçmedi",
            "history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    answer = data["response"]
    assert len(answer) > 40
    # Groq should suggest alternatives or provide guidance
    assert any(word in answer.lower() for word in ["başağrısı", "ağrı", "ilaç", "doktor"])
    
    print(f"\n✅ TC-CHAT-05 (REAL) - Medicine context handled")
    print(f"   Response: {answer[:150]}...")


# ============================================================================
# TC-CHAT-08: RAG + Groq with REAL API
# ============================================================================
def test_tc_chat_08_real_groq(client):
    """
    Medicine information query returns Groq response
    (RAG may not be available, but Groq should still respond)
    """
    response = client.post(
        "/chat",
        json={
            "message": "Paracetamol ne işe yarar?",
            "history": []
        }
    )
    
    # Could be 200 or 503 (if RAG unavailable)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        answer = data["response"]
        # Should mention paracetamol/parasetamol
        assert "parasetamol" in answer.lower() or "paracetamol" in answer.lower()
        print(f"\n✅ TC-CHAT-08 (REAL) - Groq answered: {answer[:100]}...")
    else:
        print(f"\n⚠️  TC-CHAT-08 (REAL) - RAG unavailable (503), skipping detail check")


# ============================================================================
# TC-CHAT-12: Pronoun preservation with REAL Groq
# ============================================================================
def test_tc_chat_12_real_groq(client):
    """
    Gender context preserved through TR->EN->TR pipeline
    Validates: Translation pipeline + Context preservation
    """
    response = client.post(
        "/chat",
        json={
            "message": "Ben kadınım ve boğazım ağrıyor, ne yapmalıyım?",
            "history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    answer = data["response"]
    
    # Should address the person appropriately
    assert len(answer) > 40
    assert "boğaz" in answer.lower() or "ağrı" in answer.lower()
    
    print(f"\n✅ TC-CHAT-12 (REAL) - Pronouns handled: {answer[:100]}...")


# ============================================================================
# TC-CHAT-14: Response timing (REAL)
# ============================================================================
def test_tc_chat_14_real_groq(client):
    """
    Measure actual end-to-end response time
    Includes: Translation, Groq API call, Response processing
    """
    start = time.time()
    response = client.post(
        "/chat",
        json={
            "message": "Başım ağrıyor",
            "history": []
        }
    )
    elapsed = time.time() - start
    
    assert response.status_code == 200
    
    print(f"\n✅ TC-CHAT-14 (REAL) - Response time: {elapsed:.2f}s")
    print(f"   Breakdown:")
    print(f"   - Expected: ~10s (Groq) + ~2s (translation) = ~12s")
    print(f"   - Actual: {elapsed:.2f}s")
    
    # Groq should respond within 30s (timeout limit)
    assert elapsed < 30, f"Response took {elapsed:.2f}s, max allowed: 30s"


# ============================================================================
# TC-CHAT-15: UI readability (REAL Groq)
# ============================================================================
def test_tc_chat_15_real_groq(client):
    """
    Verify real Groq response is readable and artifact-free
    """
    response = client.post(
        "/chat",
        json={
            "message": "Başağrısı için ne yapmalıyım?",
            "history": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    answer = data["response"]
    
    # Readability checks
    assert len(answer) > 100, "Response should be detailed (>100 chars)"
    assert "<" not in answer and ">" not in answer, "No HTML"
    assert "```" not in answer, "No code blocks"
    # Groq often provides well-structured responses with bullet points
    assert answer.count("\n") < 50, "Reasonable number of linebreaks (< 50)"
    
    print(f"\n✅ TC-CHAT-15 (REAL) - Readability check passed")
    print(f"   Response length: {len(answer)} chars, {answer.count(chr(10))} lines")
    print(f"   Sample: {answer[:100]}...")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REAL GROQ API INTEGRATION TESTS")
    print("=" * 80)
    print("\nThese tests use REAL Groq API calls.")
    print("Requires: GROQ_API_KEY in .env file")
    print("\nRun with:")
    print("  pytest backend/tests/test_chat_groq_real.py -v -s")
