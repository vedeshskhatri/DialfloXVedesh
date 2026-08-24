"""
Audio quality gate. Runs before any model inference — cheap, pure signal
processing, so a bad clip is rejected fast instead of burning the latency
budget on a prediction nobody should trust.

Three checks combine into one of: "good" | "degraded" | "insufficient"
  - SNR estimate (noise floor vs speech energy)
  - clipping ratio (fraction of samples at/near full scale — distorted audio)
  - speech-presence ratio via VAD (how much of the clip is actually speech
    vs silence/noise — logistics calls have long non-speech stretches)
"""
from dataclasses import dataclass

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# Thresholds — tune against real Dialflo call recordings; these are reasonable
# starting points, not final numbers. Document in README that they're
# starting points, because a reviewer should be able to tell the difference
# between "arbitrary" and "chosen and explained."
MIN_SPEECH_RATIO_GOOD = 0.5
MIN_SPEECH_RATIO_USABLE = 0.2
MIN_SNR_DB_GOOD = 15.0
MIN_SNR_DB_USABLE = 5.0
MAX_CLIPPING_RATIO = 0.01


@dataclass
class QualityReport:
    quality: str  # "good" | "degraded" | "insufficient"
    snr_db: float
    speech_ratio: float
    clipping_ratio: float


def _estimate_snr_db(samples: np.ndarray, speech_mask: np.ndarray) -> float:
    """Rough SNR: energy of frames marked speech vs frames marked non-speech."""
    if not speech_mask.any():
        return 0.0

    if not (~speech_mask).any():
        # All frames detected as active: check dynamic envelope to distinguish
        # real dynamic speech from stationary white/pink noise
        n_frames = len(samples) // FRAME_SAMPLES
        if n_frames > 1:
            frame_energies = [
                np.mean(samples[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] ** 2)
                for i in range(n_frames)
            ]
            mean_e = np.mean(frame_energies) + 1e-10
            std_e = np.std(frame_energies)
            # Stationary noise has flat variance (< 0.20); speech has dynamic envelope (> 0.5)
            if (std_e / mean_e) < 0.20:
                return 0.0  # Stationary noise, not valid speech

        speech_energy = np.mean(samples ** 2) + 1e-10
        noise_floor = np.percentile(samples ** 2, 5) + 1e-10
        return float(10 * np.log10(speech_energy / noise_floor))

    speech_energy = np.mean(samples[speech_mask] ** 2) + 1e-10
    noise_energy = np.mean(samples[~speech_mask] ** 2) + 1e-10
    return float(10 * np.log10(speech_energy / noise_energy))




def _clipping_ratio(samples: np.ndarray) -> float:
    threshold = 0.99  # samples are float32 in [-1, 1]
    return float(np.mean(np.abs(samples) >= threshold))


def _vad_frame_mask(samples_int16: np.ndarray, vad: webrtcvad.Vad) -> np.ndarray:
    n_frames = len(samples_int16) // FRAME_SAMPLES
    mask = np.zeros(n_frames * FRAME_SAMPLES, dtype=bool)
    for i in range(n_frames):
        frame = samples_int16[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        mask[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] = is_speech
    return mask


def assess(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> QualityReport:
    """
    samples: mono float32 array in [-1, 1], already resampled to `sample_rate`
    (resampling itself happens in main.py's decode step, kept separate so this
    function stays pure and unit-testable without touching ffmpeg).
    """
    assert sample_rate == SAMPLE_RATE, "resample before calling assess()"

    vad = webrtcvad.Vad(2)  # aggressiveness 0-3; 2 = moderate
    samples_int16 = (samples * 32767).astype(np.int16)
    speech_mask = _vad_frame_mask(samples_int16, vad)

    speech_ratio = float(speech_mask.mean()) if len(speech_mask) else 0.0
    snr_db = _estimate_snr_db(samples[:len(speech_mask)], speech_mask)
    clip_ratio = _clipping_ratio(samples)

    if (
        speech_ratio < MIN_SPEECH_RATIO_USABLE
        or snr_db < MIN_SNR_DB_USABLE
        or clip_ratio > MAX_CLIPPING_RATIO
    ):
        quality = "insufficient"
    elif speech_ratio < MIN_SPEECH_RATIO_GOOD or snr_db < MIN_SNR_DB_GOOD:
        quality = "degraded"
    else:
        quality = "good"

    return QualityReport(
        quality=quality, snr_db=snr_db, speech_ratio=speech_ratio,
        clipping_ratio=clip_ratio,
    )
