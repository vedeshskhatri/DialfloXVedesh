import wave
import numpy as np

# Generate a 4-second speech-like harmonic sample at 16kHz
sr = 16000
duration = 4.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Synthesize a voiced vowel sound (fundamental ~140Hz + Formants at 700Hz, 1200Hz, 2500Hz)
f0 = 140.0
signal = 0.4 * np.sin(2 * np.pi * f0 * t)
signal += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
signal += 0.20 * np.sin(2 * np.pi * 700 * t)
signal += 0.15 * np.sin(2 * np.pi * 1200 * t)
signal += 0.10 * np.sin(2 * np.pi * 2500 * t)

# Modulate amplitude smoothly to mimic natural speech syllables
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
signal = signal * envelope

# Normalize to int16 range
signal = np.clip(signal, -1.0, 1.0)
signal_int16 = (signal * 32767 * 0.7).astype(np.int16)

with wave.open("samples/sample.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(signal_int16.tobytes())

print("Created samples/sample.wav successfully!")
