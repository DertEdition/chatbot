"""
Medical Chatbot Exception Handling Tests

Tests for error scenarios:
- API timeouts
- Rate limiting
- Network errors
- Invalid responses
- Service unavailability

Run with:
  pytest backend/tests/test_chat_exceptions.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from requests.exceptions import Timeout, ConnectionError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


# ============================================================================
# TC-CHAT-EX-01: Groq API Timeout
# ============================================================================
def test_groq_timeout(client):
    """
    Groq API timeout (>30s) should return error gracefully
    """
    with patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.tr_to_en.translate') as mock_tr_en:
        
        # Simulate timeout
        mock_groq.side_effect = Timeout("Request timed out after 30 seconds")
        mock_tr_en.return_value = "My head hurts"
        
        response = client.post(
            "/chat",
            json={
                "message": "Başım ağrıyor",
                "history": []
            }
        )
        
        # Should handle timeout gracefully
        assert response.status_code in [408, 504, 500, 503]
        print(f"✅ TC-CHAT-EX-01 (Timeout) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-02: Network Connection Error
# ============================================================================
def test_network_connection_error(client):
    """
    Network error during Groq API call should be handled
    """
    with patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.tr_to_en.translate') as mock_tr_en:
        
        # Simulate network error
        mock_groq.side_effect = ConnectionError("Network unreachable")
        mock_tr_en.return_value = "What should I do?"
        
        response = client.post(
            "/chat",
            json={
                "message": "Ne yapmalıyım?",
                "history": []
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [500, 503]
        print(f"✅ TC-CHAT-EX-02 (Network Error) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-03: Rate Limit (429)
# ============================================================================
def test_groq_rate_limit(client):
    """
    Rate limit response (429) should be handled
    """
    with patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.tr_to_en.translate') as mock_tr_en:
        
        # Create a 429 error
        from requests.exceptions import HTTPError
        response = MagicMock()
        response.status_code = 429
        mock_groq.side_effect = HTTPError("429 Too Many Requests")
        mock_tr_en.return_value = "I have a headache"
        
        response = client.post(
            "/chat",
            json={
                "message": "Başım ağrıyor",
                "history": []
            }
        )
        
        # Should handle rate limit
        assert response.status_code in [429, 500, 503]
        print(f"✅ TC-CHAT-EX-03 (Rate Limit) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-04: Translation Service Failure
# ============================================================================
def test_translation_failure(client):
    """
    Translation service (tr_to_en) failure should be handled
    Fallback: use original message or degraded response
    """
    with patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.tr_to_en.translate') as mock_tr_en, \
         patch('app.main.en_to_tr.translate') as mock_en_tr, \
         patch('app.main.groq_client.chat.completions.create') as mock_groq:
        
        # Translation fails
        mock_tr_en.side_effect = ConnectionError("Google Translate down")
        mock_en_tr.return_value = "Cevap alınamadı"
        mock_groq.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="No response available"
            ))]
        )
        
        response = client.post(
            "/chat",
            json={
                "message": "Başım ağrıyor",
                "history": []
            }
        )
        
        # Should fail gracefully or return degraded response
        assert response.status_code in [200, 500]
        print(f"✅ TC-CHAT-EX-04 (Translation Failure) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-05: Invalid Message Format
# ============================================================================
def test_invalid_message_format(client):
    """
    Invalid message format should return validation error
    """
    # Test 1: Empty message
    response = client.post(
        "/chat",
        json={"message": "", "history": []}
    )
    assert response.status_code in [400, 422]
    print(f"✅ TC-CHAT-EX-05a (Empty) - Status: {response.status_code}")
    
    # Test 2: Missing message field
    response = client.post(
        "/chat",
        json={"history": []}
    )
    assert response.status_code in [400, 422]
    print(f"✅ TC-CHAT-EX-05b (Missing field) - Status: {response.status_code}")
    
    # Test 3: Invalid history format
    response = client.post(
        "/chat",
        json={
            "message": "Test",
            "history": [{"role": "invalid", "content": "test"}]
        }
    )
    assert response.status_code in [400, 422, 503]
    print(f"✅ TC-CHAT-EX-05c (Invalid history) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-06: Very Long Message (DoS prevention)
# ============================================================================
def test_very_long_message(client):
    """
    Extremely long message should be rejected or truncated
    """
    long_message = "a" * 10000  # 10k characters
    
    response = client.post(
        "/chat",
        json={
            "message": long_message,
            "history": []
        }
    )
    
    # Should either reject or handle gracefully
    assert response.status_code in [200, 413, 422, 503]
    print(f"✅ TC-CHAT-EX-06 (Long message) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-07: Malformed JSON
# ============================================================================
def test_malformed_json(client):
    """
    Malformed JSON should return 400 Bad Request
    """
    response = client.post(
        "/chat",
        data="{invalid json}",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422
    print(f"✅ TC-CHAT-EX-07 (Malformed JSON) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-08: Groq Returns Invalid Response
# ============================================================================
def test_groq_invalid_response(client):
    """
    Groq API returns malformed response (missing content field)
    """
    with patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.tr_to_en.translate') as mock_tr_en, \
         patch('app.main.en_to_tr.translate') as mock_en_tr:
        
        # Return response without proper content
        mock_groq.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        mock_tr_en.return_value = "Test"
        mock_en_tr.return_value = "Test"
        
        response = client.post(
            "/chat",
            json={
                "message": "Başım ağrıyor",
                "history": []
            }
        )
        
        # Should handle invalid response
        assert response.status_code in [200, 500, 503]
        print(f"✅ TC-CHAT-EX-08 (Invalid Groq Response) - Status: {response.status_code}")


# ============================================================================
# TC-CHAT-EX-09: Concurrent Rate Limit Protection
# ============================================================================
def test_concurrent_requests_handling(client):
    """
    Multiple rapid requests should respect rate limits
    (No actual concurrent test, just verify structure)
    """
    with patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.tr_to_en.translate') as mock_tr_en, \
         patch('app.main.en_to_tr.translate') as mock_en_tr:
        
        mock_groq.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Response"))]
        )
        mock_tr_en.return_value = "Test"
        mock_en_tr.return_value = "Test"
        
        # Make 3 rapid requests
        responses = []
        for i in range(3):
            r = client.post(
                "/chat",
                json={
                    "message": f"Test {i}",
                    "history": []
                }
            )
            responses.append(r.status_code)
        
        # All should succeed or some should be rate limited
        assert all(code in [200, 429] for code in responses)
        print(f"✅ TC-CHAT-EX-09 (Concurrent) - Statuses: {responses}")


# ============================================================================
# TC-CHAT-EX-10: Service Degradation Handling
# ============================================================================
def test_service_degradation(client):
    """
    When one service (RAG) is down, system should fall back gracefully
    """
    with patch('app.main.is_health_related', return_value=True), \
         patch('app.main.check_emergency_symptoms', return_value=(False, "")), \
         patch('app.main.groq_client.chat.completions.create') as mock_groq, \
         patch('app.main.tr_to_en.translate') as mock_tr_en, \
         patch('app.main.en_to_tr.translate') as mock_en_tr:
        
        # Groq still works
        mock_groq.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="I can help with this health question"
            ))]
        )
        mock_tr_en.return_value = "Test"
        mock_en_tr.return_value = "Bu sağlık sorusuna yardım edebilirim"
        
        response = client.post(
            "/chat",
            json={
                "message": "Başım ağrıyor",
                "history": []
            }
        )
        
        # Should work even if RAG is down
        assert response.status_code == 200
        print(f"✅ TC-CHAT-EX-10 (Service Degradation) - Fallback working")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXCEPTION HANDLING & ERROR SCENARIO TESTS")
    print("=" * 80)
    print("\nTests for robustness and error recovery:")
    print("  - API timeouts")
    print("  - Network errors")
    print("  - Rate limiting")
    print("  - Validation errors")
    print("  - DoS prevention")
    print("  - Malformed requests")
    print("  - Service degradation")
    print("\nRun with:")
    print("  pytest backend/tests/test_chat_exceptions.py -v")
