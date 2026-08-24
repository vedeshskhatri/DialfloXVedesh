"""
Integration smoke test for the FastAPI app — no real model needed.

We mock inference.predict() so this test:
  1. Is fast (< 1s)
  2. Works without downloading the 1.2 GB model weights
  3. Still exercises the full HTTP path end-to-end: upload → pipeline → response

Run:
  pytest tests/test_api_smoke.py -v

With the real model (slow, requires weights):
  DIALFLO_REAL_MODEL=1 pytest tests/test_api_smoke.py -v
"""
import io
import os
import sys
import struct
import wave
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixture: mock inference so tests run without downloading model weights
# ---------------------------------------------------------------------------

def _make_mock_prediction(gender="male", gender_conf=0.85,
                           age_bracket="31-45", age_conf=0.72):
    from app.inference import RawPrediction
    return RawPrediction(
        gender_label=gender,
        gender_confidence=gender_conf,
        age_bracket=age_bracket,
        age_confidence=age_conf,
        language_label="en-IN",
        language_confidence=0.78,
    )


USE_REAL_MODEL = os.environ.get("DIALFLO_REAL_MODEL", "0") == "1"


@pytest.fixture
def client():
    """TestClient with inference mocked unless DIALFLO_REAL_MODEL=1."""
    from app.main import app
    if USE_REAL_MODEL:
        yield TestClient(app)
    else:
        mock_pred = _make_mock_prediction()
        with patch("app.inference.predict", return_value=mock_pred):
            yield TestClient(app)


# ---------------------------------------------------------------------------
# Helper: generate a minimal valid WAV file in memory
# ---------------------------------------------------------------------------

def _make_wav(duration_s: float = 3.0, freq_hz: float = 300.0,
              sample_rate: int = 16_000) -> bytes:
    """Generate a clean sine-wave WAV in memory (no disk I/O)."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples_f32 = (0.4 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    samples_i16 = (samples_f32 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_returns_200(client):
    wav_bytes = _make_wav(3.0)
    resp = client.post(
        "/analyze",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 200


def test_analyze_response_shape(client):
    """Response must match the contract exactly."""
    wav_bytes = _make_wav(3.0)
    resp = client.post(
        "/analyze",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    body = resp.json()

    # Required top-level fields
    for field in ("contact_id", "gender", "age_bracket", "processing_ms",
                  "audio_quality", "decision"):
        assert field in body, f"Missing field: {field}"

    # Gender shape
    assert "prediction" in body["gender"]
    assert "confidence" in body["gender"]
    assert body["gender"]["prediction"] in ("male", "female", "unknown")
    assert 0.0 <= body["gender"]["confidence"] <= 1.0

    # Age bracket shape
    assert "prediction" in body["age_bracket"]
    assert body["age_bracket"]["prediction"] in ("18-30", "31-45", "46-60", "60+", "unknown")

    # audio_quality enum
    assert body["audio_quality"] in ("good", "degraded", "insufficient")

    # decision enum (our extension)
    assert body["decision"] in ("auto_use", "flag_for_review", "discard")

    # processing_ms is a non-negative integer
    assert isinstance(body["processing_ms"], int)
    assert body["processing_ms"] >= 0


def test_analyze_with_contact_id(client):
    wav_bytes = _make_wav(3.0)
    resp = client.post(
        "/analyze",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        data={"contact_id": "test-contact-123"},
    )
    assert resp.status_code == 200
    assert resp.json()["contact_id"] == "test-contact-123"


def test_analyze_without_contact_id_generates_uuid(client):
    wav_bytes = _make_wav(3.0)
    resp = client.post(
        "/analyze",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    body = resp.json()
    # Should auto-generate a contact_id
    assert body["contact_id"] is not None
    assert len(body["contact_id"]) > 0


def test_silence_returns_insufficient(client):
    """Zero-amplitude WAV → quality gate → insufficient response."""
    n_samples = 16_000 * 3
    samples_i16 = np.zeros(n_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(samples_i16.tobytes())
    silent_wav = buf.getvalue()

    resp = client.post(
        "/analyze",
        files={"audio": ("silent.wav", silent_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["audio_quality"] == "insufficient"
    assert body["decision"] == "discard"
    assert body["gender"]["prediction"] == "unknown"
    assert body["age_bracket"]["prediction"] == "unknown"


def test_corrupt_audio_returns_200_not_500(client):
    """Corrupt audio bytes must never cause a 500 — always return the contract shape."""
    resp = client.post(
        "/analyze",
        files={"audio": ("corrupt.wav", b"not-a-real-audio-file", "audio/wav")},
    )
    # Must be 200 with a valid contract response (not a 500 with a traceback)
    assert resp.status_code == 200
    body = resp.json()
    assert "gender" in body
    assert "age_bracket" in body


def test_short_clip_returns_insufficient(client):
    """Clips shorter than MIN_USABLE_SECONDS → insufficient without running model."""
    # 0.3s — well below the 1.0s minimum
    wav_bytes = _make_wav(0.3)
    resp = client.post(
        "/analyze",
        files={"audio": ("short.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["audio_quality"] == "insufficient"
