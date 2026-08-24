"""
Real model verification and benchmark test.
Only runs when DIALFLO_REAL_MODEL=1 is set in environment or when executed directly.
"""
import os
import pytest
import numpy as np
import torch

from app.inference import predict, _load_model, AgeGenderModel

@pytest.mark.skipif(
    os.environ.get("DIALFLO_REAL_MODEL") != "1",
    reason="Requires full 1.2GB model weights download. Run with DIALFLO_REAL_MODEL=1 to execute."
)
def test_real_model_inference():
    processor, model = _load_model()
    assert isinstance(model, AgeGenderModel)
    
    # 3-second random test waveform at 16kHz
    signal = np.random.randn(48000).astype(np.float32) * 0.1
    pred = predict(signal, sample_rate=16000)
    
    assert pred.gender_label in ("male", "female", "unknown")
    assert 0.0 <= pred.gender_confidence <= 1.0
    assert pred.age_bracket in ("18-30", "31-45", "46-60", "60+")
    assert 0.0 <= pred.age_confidence <= 1.0
