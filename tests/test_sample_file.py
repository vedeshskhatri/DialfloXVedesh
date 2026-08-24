import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.inference import RawPrediction

def test_sample_wav_file_exists():
    assert os.path.exists("samples/sample.wav")
    assert os.path.getsize("samples/sample.wav") > 1000

def test_analyze_with_sample_wav():
    client = TestClient(app)
    with open("samples/sample.wav", "rb") as f:
        audio_bytes = f.read()
    
    mock_pred = RawPrediction(
        gender_label="male",
        gender_confidence=0.88,
        age_bracket="31-45",
        age_confidence=0.75,
        language_label="en-IN",
        language_confidence=0.80,
    )
    
    with patch("app.inference.predict", return_value=mock_pred):
        response = client.post(
            "/analyze",
            files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
            data={"contact_id": "dialflo-driver-test-01"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["contact_id"] == "dialflo-driver-test-01"
    assert "gender" in data
    assert "age_bracket" in data
    assert "audio_quality" in data
    assert "decision" in data
    assert "processing_ms" in data
    print("\nSample inference result:", data)
