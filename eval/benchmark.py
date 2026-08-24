import time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from app.inference import predict, MODEL_ID, _load_model

print(f"Downloading/Verifying model {MODEL_ID}...")
t0 = time.time()
path = snapshot_download(repo_id=MODEL_ID)
print(f"Downloaded in {time.time() - t0:.2f}s to {path}")

print("Loading model...")
t0 = time.time()
processor, model = _load_model()
print(f"Model loaded in {time.time() - t0:.2f}s")

# 5-second test audio at 16kHz
t = np.linspace(0, 5, 80000, endpoint=False)
# Generate modulated speech-like audio (mix of fundamental frequency 130Hz + harmonics)
sample_audio = (
    0.3 * np.sin(2 * np.pi * 130 * t) +
    0.2 * np.sin(2 * np.pi * 260 * t) +
    0.1 * np.sin(2 * np.pi * 390 * t) +
    0.05 * np.random.randn(len(t))
).astype(np.float32)

print("Running warmup inference on 5s clip...")
pred = predict(sample_audio, sample_rate=16000)
print("Warmup Prediction:", pred)

print("Benchmarking 5 iterations...")
latencies = []
for i in range(5):
    t0 = time.perf_counter()
    pred = predict(sample_audio, sample_rate=16000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    latencies.append(elapsed_ms)
    print(f"  Run {i+1}: {elapsed_ms:.1f}ms -> Gender: {pred.gender_label} ({pred.gender_confidence}), Age: {pred.age_bracket} ({pred.age_confidence})")

avg_latency = np.mean(latencies)
print(f"\nAverage 5-second inference latency on CPU: {avg_latency:.1f}ms")
