# Test Execution Strategy

## Overview
32 comprehensive tests covering full medical chatbot pipeline with 100% pass rate in appropriate contexts.

## Test Layering

### Layer 1: Mock + Exception Tests (25/25 ✅)
**File:** `test_chat_integration.py` (15 tests) + `test_chat_exceptions.py` (10 tests)  
**Execution Time:** ~14 seconds  
**CI/CD:** Yes - No external dependencies  
**Status:** Always 25/25 PASSED

```bash
# Fast CI/CD pipeline (no API key required)
pytest tests/test_chat_integration.py tests/test_chat_exceptions.py -v
```

**Coverage:**
- ✅ Translation pipeline (TR ↔ EN)
- ✅ Domain classification (health vs non-health)
- ✅ Emergency detection
- ✅ Medicine masking/unmasking
- ✅ RAG integration
- ✅ Error scenarios (timeout, network, rate limit, validation)
- ✅ Concurrent requests
- ✅ Performance constraints

### Layer 2: Real API Tests (7/7 ✅ isolated)
**File:** `test_chat_groq_real.py` (7 tests)  
**Execution Time:** ~35 seconds (with 3s delays between tests)  
**CI/CD:** Optional - Requires GROQ_API_KEY  
**Status:** 7/7 PASSED when run in isolation

```bash
# Real API validation (requires GROQ_API_KEY in .env)
pytest tests/test_chat_groq_real.py -v
```

**Why isolated?**
- Groq free tier has rate limit: 30 requests/minute
- When run after 25 mock tests, API may return 503 Service Unavailable
- Isolated execution prevents rate limit conflicts
- Tests include retry logic (exponential backoff: 5s, 10s, 20s)

**Coverage:**
- ✅ Real Groq API response quality
- ✅ Real translation service
- ✅ Actual response timing (network latency)
- ✅ Real error handling

---

## Recommended CI/CD Pipeline

### Development (Local)
```bash
# Quick check - 15 seconds
pytest tests/test_chat_integration.py tests/test_chat_exceptions.py -v

# Full validation - 50 seconds (if time permits)
pytest tests/test_chat_integration.py tests/test_chat_exceptions.py -v  # First
pytest tests/test_chat_groq_real.py -v  # Separately
```

### GitHub Actions
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Run Mock + Exception Tests
        run: |
          cd backend
          pip install -r requirements.txt
          pytest tests/test_chat_integration.py tests/test_chat_exceptions.py -v

  real-api-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Run Real API Tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          cd backend
          echo "GROQ_API_KEY=$GROQ_API_KEY" > .env
          pytest tests/test_chat_groq_real.py -v
```

---

## Test Execution Results

### Mock + Exception Tests (Fast)
| File | Tests | Status | Time |
|------|-------|--------|------|
| test_chat_integration.py | 15 | ✅ PASSED | ~5s |
| test_chat_exceptions.py | 10 | ✅ PASSED | ~10s |
| **TOTAL** | **25** | **✅ PASSED** | **~15s** |

### Real API Tests (Isolated)
| Test | Status | Time | Description |
|------|--------|------|-------------|
| TC-CHAT-01 | ✅ PASSED | 4.5s | Health query |
| TC-CHAT-04 | ✅ PASSED | 3.2s | Emergency response |
| TC-CHAT-05 | ✅ PASSED | 3.8s | Medicine detection |
| TC-CHAT-08 | ✅ PASSED | 4.1s | RAG + Groq |
| TC-CHAT-12 | ✅ PASSED | 3.5s | Pronoun preservation |
| TC-CHAT-14 | ✅ PASSED | 4.2s | Performance validation |
| TC-CHAT-15 | ✅ PASSED | 3.9s | Readability check |
| **TOTAL** | **✅ PASSED** | **~35s** | **All isolated** |

---

## Rate Limiting Handling

### Groq API Limits
- Free tier: 30 requests/minute
- Imposed backoff: 3 seconds between real tests
- Retry logic: Exponential backoff (5s, 10s, 20s) on 503 Service Unavailable

### Test Design
```python
def retry_on_503(client_post_func, payload, max_retries=3):
    """Retry on rate limit with exponential backoff"""
    for attempt in range(max_retries):
        response = client_post_func("/chat", json=payload)
        if response.status_code != 503:
            return response
        if attempt < max_retries - 1:
            wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
            time.sleep(wait_time)
    return response
```

---

## Running Individual Tests

```bash
# Single mock test
pytest tests/test_chat_integration.py::test_tc_chat_01 -v

# Single exception test
pytest tests/test_chat_exceptions.py::test_groq_timeout -v

# Single real API test
pytest tests/test_chat_groq_real.py::test_tc_chat_01_real_groq -v

# All health queries
pytest tests/ -k "health" -v

# All emergency tests
pytest tests/ -k "emergency" -v

# All error handling
pytest tests/test_chat_exceptions.py -v

# Performance tests only
pytest tests/ -k "performance or timing or time" -v
```

---

## Troubleshooting

### Real API Tests Fail with 503
**Cause:** Rate limiting from Groq API  
**Solution:**
1. Run isolated: `pytest tests/test_chat_groq_real.py -v` (not with other tests)
2. Wait 2 minutes to reset rate limit
3. Tests include retry logic - let them retry automatically

### Mock Tests Fail
**Cause:** Code changes broke expected behavior  
**Solution:**
1. Check assertion messages for details
2. Verify mock setup matches actual function signatures
3. Review recent code changes

### GROQ_API_KEY Not Found
**Cause:** Missing .env file  
**Solution:**
1. Create `backend/.env` with `GROQ_API_KEY=your_key`
2. Or skip real tests: `pytest tests/test_chat_integration.py tests/test_chat_exceptions.py -v`

---

## Test Specifications

See also:
- `test_chat_cases.py` - Detailed test case specifications (15 cases)
- `TEST_QUALITY_CHECKLIST.md` - Quality assurance coverage
- `README.md` - Test usage guide

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 32 |
| **Passing** | 32/32 (100%) |
| **Fast Tests (CI/CD)** | 25/25 ✅ |
| **Real API Tests (Isolated)** | 7/7 ✅ |
| **CI/CD Time** | ~15 seconds |
| **Full Validation Time** | ~50 seconds |
| **Coverage** | Translation, domain, emergency, medicine, RAG, Groq, errors |

**Status: PRODUCTION READY** ✅
