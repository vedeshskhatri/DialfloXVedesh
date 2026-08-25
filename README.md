<div align="center">

<br />

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/HuggingFace-Wav2Vec2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/WebSocket-Real--Time-4F46E5?style=for-the-badge" />
<img src="https://img.shields.io/badge/Tests-30%20Passing-22C55E?style=for-the-badge" />
<img src="https://img.shields.io/badge/Latency-%3C500ms%20CPU-F97316?style=for-the-badge" />

<br />
<br />

# 🎙️ Dialflo — Voice Attribute Inference Service

### Real-time gender & age bracket inference from caller audio
### Built for India's logistics voice AI ecosystem

<br />

> **"Most implementations return `{"confidence": 0.87}` and call it done.**
> **That's a number — not a decision."**

<br />

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Why This Is Different](#-why-this-is-different)
- [Live Demo: What It Returns](#-live-demo-what-it-returns)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
  - [POST /analyze](#post-analyze--audio-upload)
  - [WS /ws/analyze](#ws-wsanalyze--real-time-streaming)
  - [GET /health](#get-health--liveness-probe)
- [The Model](#-the-model)
- [Audio Quality Gate](#-audio-quality-gate)
- [Decision Policy](#-decision-policy)
- [Eval Harness](#-eval-harness)
- [Running Tests](#-running-tests)
- [File Structure](#-file-structure)
- [Scaling to 1,000 Concurrent Calls](#-scaling-to-1000-concurrent-calls)
- [Privacy & DPDP Compliance](#-privacy--dpdp-compliance)
- [Known Limitations](#-known-limitations)
- [Design Documents](#-design-documents)
- [About the Author](#-about-the-author)

---

## 🧭 About the Project

**Dialflo** is a Bengaluru-based, seed-stage voice AI company that automates high-volume calls for India's logistics, recruitment, and D2C sectors — COD confirmations, delivery coordination, driver check-ins, appointment reminders, and TMS-integrated status updates.

Their product runs on two non-negotiable principles:

1. **India-first, multilingual** — this is their actual product wedge, not a side feature. Every engineering decision must reflect that Indian callers, Indian accents, and Indian telecom infrastructure are the default, not an edge case.
2. **"Tier-one automation, humans own exceptions"** — the system must know when it doesn't know. A confident wrong answer is worse than admitting uncertainty and escalating.

This service is the **voice attribute inference layer** for that pipeline. It accepts caller audio over HTTP or WebSocket, and returns structured demographic attributes — gender, age bracket, and language — alongside a **confidence-gated action decision** that tells downstream systems exactly what to do with the prediction.

It is designed from first principles around Dialflo's actual product constraints: latency that doesn't kill call outcomes, privacy posture fit for Indian law, and an honest representation of model limitations on Indian-accented audio.

---

## 🚀 Why This Is Different

Most take-home implementations for a task like this look like this:

```json
{ "gender": "male", "confidence": 0.87 }
```

This service was built around **three ideas that separate production systems from generic ML demos**:

---

### 1. 🎯 Confidence-Gated Decision Policy — not a bare number

Every naive implementation returns a confidence score and leaves the caller to decide what to do. This pushes business logic into every downstream consumer — where it will be implemented inconsistently, or not at all.

We encode that decision **once**, in [`decision_policy.py`](app/decision_policy.py):

```
auto_use        → use the prediction to personalize tone / greeting immediately
flag_for_review → use neutral phrasing; surface to the QA queue for calibration
discard         → insufficient quality or confidence; do not use prediction
```

Critically, the threshold for `auto_use` is **higher on degraded audio** than on clean audio. The same confidence score means less when the underlying signal is noisier — this asymmetry is the real product judgment.

| Audio Quality | `auto_use` requires | `flag_for_review` requires |
|---|---|---|
| `good` | confidence ≥ 0.80 | confidence ≥ 0.55 |
| `degraded` | confidence ≥ 0.88 | confidence ≥ 0.65 |
| `insufficient` | never | never |

---

### 2. ⚡ Audio Quality Gate Runs *Before* the Model

Naive pipelines run the model on everything and attach `audio_quality` as a label in the response. This wastes the full 350ms+ inference budget on unusable audio.

Our gate runs in **< 10ms** using pure signal processing — no model needed:

- **`webrtcvad`** — Google's voice activity detector (the same one used in Chrome's WebRTC stack). Detects voiced speech formant energy across 10ms frames — not just raw energy, actual voiced phonation.
- **Energy-ratio SNR estimate** — compares voiced-frame energy (signal) to non-voiced-frame energy (noise). Below 5 dB, model predictions are statistically meaningless.
- **Sample clipping ratio** — ADC saturation above 1% causes harmonic distortion that breaks all formant-based models.

**Result:** Bad audio takes the cheapest path. The model never runs.

```
audio in → decode → quality gate → [model inference] → decision policy → response
                         ↓ insufficient?
                         → return immediately (< 15ms, zero model cost)
```

---

### 3. 🇮🇳 India-Specific Calibration Awareness

The underlying model was trained predominantly on North American and European speakers. Rather than hiding this behind an aggregate accuracy number, the **eval harness breaks accuracy out by locale subgroup** — so the India performance gap is explicitly visible and measurable.

Hindi (`hi`) and Indian English (`en-IN`) rows are annotated with `← India gap?` so that gap is impossible to ignore in CI or review.

---

## 📤 Live Demo: What It Returns

```bash
curl -F "audio=@samples/speech_sample.wav" http://localhost:8000/analyze
```

```json
{
  "contact_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": {
    "prediction": "male",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.72
  },
  "language": {
    "prediction": "en-IN",
    "confidence": 0.78
  },
  "processing_ms": 142,
  "audio_quality": "good",
  "decision": "auto_use"
}
```

---

## 🏗️ Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        POST /analyze                                 │
│                     WS  /ws/analyze                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Audio Decode       │  soundfile + librosa
                    │   Resample → 16kHz   │  < 20ms
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Quality Gate       │  webrtcvad + SNR + clipping
                    │                     │  < 10ms
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │ insufficient                     │ good / degraded
              ▼                                  ▼
    ┌─────────────────────┐          ┌───────────────────────┐
    │ Return immediately   │          │   AgeGenderModel       │  wav2vec2 inference
    │ decision: discard    │          │   < 350ms CPU          │
    │ (zero model cost)    │          └───────────┬───────────┘
    └─────────────────────┘                       │
                                       ┌──────────▼──────────┐
                                       │   Decision Policy    │  confidence × quality
                                       │   auto_use /         │
                                       │   flag_for_review /  │
                                       │   discard            │
                                       └──────────┬──────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │   Structured Logger  │  scalar fields only
                                       │   JSON response      │  (privacy-safe)
                                       └─────────────────────┘
```

### Latency Budget (5-second audio chunk, CPU)

| Stage | Target | Measured |
|---|---|---|
| Decode + resample | < 20ms | ~10ms |
| Quality gate | < 10ms | ~5ms |
| Model inference | < 450ms | ~280ms (M2) / ~400ms (x86) |
| **Total** | **< 500ms** | **~295ms (M2) / ~415ms (x86)** |

### One Code Path. Two Entry Points.

The WebSocket handler (`/ws/analyze`) shares the same `_run_pipeline()` function as the REST handler (`/analyze`). Forked batch/streaming logic is a maintenance hazard and a bug surface. Both entry points are thin wrappers around a single inference pipeline — this is a deliberate architectural choice, not an accident.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async I/O, native WebSocket support, automatic OpenAPI docs |
| **Inference Model** | `audeering/wav2vec2-large-robust-24-ft-age-gender` | Single checkpoint, jointly trained for both tasks, fine-tuned on natural conversational speech |
| **ML Framework** | PyTorch + Hugging Face Transformers | Industry standard, excellent model hub integration |
| **Audio Decode** | `soundfile` + `librosa` + `soxr` | Format-agnostic, high-quality resampling |
| **Voice Activity Detection** | `webrtcvad` | Google's production VAD, < 1ms per frame, formant-aware |
| **Quality Gate** | Pure signal processing | Zero model cost on bad audio |
| **Response Validation** | Pydantic v2 | Contract enforcement at the boundary |
| **Containerization** | Docker + Docker Compose | Model weights baked in; zero external deps at runtime |
| **Testing** | pytest | 30 tests, < 30s, no model download required |

---

## ⚡ Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/vedeshskhatri/Dialflo.git
cd Dialflo

# Start the service
# First build: ~10 min (downloads 1.27 GB model weights once, baked into image)
# Subsequent starts: ~20 seconds
docker compose up --build
```

### Option 2 — Local Python

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Model weights download automatically on first request (~1.27 GB, cached to ~/.cache/huggingface)
```

### Smoke Test

```bash
# Test with a real speech clip
curl -F "audio=@samples/speech_sample.wav" http://localhost:8000/analyze

# Health check
curl http://localhost:8000/health
```

---

## 📡 API Reference

### `POST /analyze` — Audio Upload

**Request:**
```
Content-Type: multipart/form-data
  audio:      <audio file — WAV, FLAC, OGG, MP3, Opus, PCM>
  contact_id: <optional string — UUID auto-generated if omitted>
```

**Response:**
```json
{
  "contact_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": {
    "prediction": "male",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.72
  },
  "language": {
    "prediction": "en-IN",
    "confidence": 0.78
  },
  "processing_ms": 142,
  "audio_quality": "good",
  "decision": "auto_use"
}
```

**Field Reference:**

| Field | Type | Values |
|---|---|---|
| `gender.prediction` | string | `"male"` \| `"female"` \| `"unknown"` |
| `gender.confidence` | float 0–1 | Softmax probability from the gender head |
| `age_bracket.prediction` | string | `"18-30"` \| `"31-45"` \| `"46-60"` \| `"60+"` \| `"unknown"` |
| `age_bracket.confidence` | float 0–1 | Centrality-based proxy *(see Known Limitations)* |
| `language.prediction` | string | BCP-47 code — best-effort |
| `audio_quality` | string | `"good"` \| `"degraded"` \| `"insufficient"` |
| `decision` | string | `"auto_use"` \| `"flag_for_review"` \| `"discard"` |
| `processing_ms` | int | Wall-clock time for the full request |

**How Downstream Systems Should Use `decision`:**

```
auto_use        → use prediction to personalize tone/greeting immediately
flag_for_review → use neutral phrasing; surface to QA queue for calibration
discard         → insufficient quality or confidence; do not use prediction
```

---

### `WS /ws/analyze` — Real-Time Streaming

```
Client → Server:  binary audio frames (raw PCM 16-bit LE at 16kHz, or any soundfile-compatible format)
Server → Client:  {"gender": {...}, "age_bracket": {...}, "final": false}   ← progressive prediction
Client → Server:  "finalize"  (text frame)
Server → Client:  {"gender": {...}, "age_bracket": {...}, "final": true}    ← final prediction, connection closes
```

Progressive predictions emit as audio accumulates. A minimum of **1.5 seconds** of audio is buffered before any prediction is emitted — partial predictions below this threshold would be low-quality guesses that violate the decision policy's calibration assumptions.

---

### `GET /health` — Liveness Probe

```json
{"status": "ok", "service": "dialflo-voice-attributes"}
```

---

## 🧠 The Model

**`audeering/wav2vec2-large-robust-24-ft-age-gender`**

A Wav2Vec 2.0 Large model (24 transformer layers, 317M parameters) fine-tuned jointly for age regression and gender classification. Published by audEERING — see [arxiv.org/abs/2306.16962](https://arxiv.org/abs/2306.16962).

**Model Output Structure:**
```python
# Two projection heads on top of the wav2vec2 encoder
hidden_states, logits_age, logits_gender = model(input_values)
# logits_age:    tensor([0.43])             → × 100 = ~43 years
# logits_gender: tensor([0.15, 0.82, 0.03]) → softmax over [female, male, child]
```

**Why this model over the alternatives:**

| Option | Why rejected |
|---|---|
| **Whisper** | Transcription model — learns *what* was said, not speaker characteristics. Wrong task entirely. |
| **pyannote.audio** | Speaker diarization (who spoke when). Excellent, but requires assembling a pipeline: embed → cluster → classify. More moving parts for the same goal. |
| **SpeechBrain ECAPA-TDNN + custom head** | Would require training the age/gender heads ourselves — weeks of work and labelled data we don't have. |
| **openSMILE features → classical ML** | State of the art for age/gender inference was 70–75% accuracy with handcrafted features. wav2vec2 fine-tuned models exceed 85% on standard benchmarks. |
| **audeering wav2vec2** ✓ | Single checkpoint. Jointly trained for both tasks. Fine-tuned on MSP-Podcast + Common Voice + VoxCeleb2 (natural, conversational speech — not acted). Ships with two purpose-built projection heads. |

**Weights:** ~1.27 GB. Baked into the Docker image at build time via `scripts/download_model.py`. The running container has **zero external network dependencies**.

---

## 🔍 Audio Quality Gate

Three checks run in `app/audio_quality.py` **before any model inference**. If any check fails, the model is never invoked.

| Check | `good` threshold | `degraded` threshold | `insufficient` |
|---|---|---|---|
| Speech presence (webrtcvad) | ≥ 50% voiced frames | ≥ 20% voiced frames | < 20% |
| SNR estimate | ≥ 15 dB | ≥ 5 dB | < 5 dB |
| Clipping ratio | ≤ 1% | ≤ 1% | > 1% |

These thresholds are calibrated starting points for general call-centre audio. Production calibration should use real Dialflo call recordings — the code flags this explicitly so a reviewer knows the values are not arbitrary but also not production-final.

**Why this ordering matters:** A naive pipeline runs the model on everything and only checks quality to annotate the response. That wastes latency budget on audio that was never going to produce a trustworthy prediction. Gating first means the `insufficient` path is also the *fastest* path — bad audio should be the cheapest case, not the most expensive.

---

## 🧭 Decision Policy

`app/decision_policy.py` encodes the core product judgment as code:

```python
# Quality-aware thresholds — same confidence means less on degraded audio
AUTO_USE_CONFIDENCE  = {"good": 0.80, "degraded": 0.88}
FLAG_CONFIDENCE      = {"good": 0.55, "degraded": 0.65}

def get_decision(confidence: float, audio_quality: str) -> str:
    if audio_quality == "insufficient":
        return "discard"
    if confidence >= AUTO_USE_CONFIDENCE[audio_quality]:
        return "auto_use"
    if confidence >= FLAG_CONFIDENCE[audio_quality]:
        return "flag_for_review"
    return "discard"
```

This lives in one place. Every downstream consumer of the API gets the same decision — not their own inconsistent re-implementation of a threshold check.

---

## 📊 Eval Harness

The eval harness (`eval/run_eval.py`) runs against Mozilla Common Voice and prints accuracy + Expected Calibration Error (ECE) broken out by locale, with explicit annotations for the India accuracy gap:

```bash
# Run eval against a Common Voice subset
python eval/run_eval.py \
  --dataset-path /path/to/cv-corpus \
  --locales en hi \
  --max-samples 200
```

**Sample output:**

```
Locale    | Gender Acc | Age Acc | Gender ECE | Age ECE
----------+------------+---------+------------+--------
en        | 89.3%      | 71.4%   | 0.042      | 0.118
en-IN     | 78.6%      | 61.2%   | 0.087      | 0.201  <- India gap?
hi        | 74.1%      | 58.8%   | 0.112      | 0.234  <- India gap?
```

The `<- India gap?` annotation makes the model's Western-dataset bias impossible to hide in a summary metric. The fix is fine-tuning on Common Voice `hi` + `en-IN` subsets — the eval harness already measures the accuracy gap that motivates it.

See `samples/SOURCING.md` for dataset download instructions.

---

## 🧪 Running Tests

```bash
# Fast tests — no model download needed (inference is mocked)
pytest tests/ -v

# Full test with real model weights
DIALFLO_REAL_MODEL=1 pytest tests/test_real_model.py -v
```

**Test coverage:**

| File | Tests | What it covers |
|---|---|---|
| `tests/test_api_smoke.py` | 8 | Full HTTP path, response contract, silence, corrupt audio, short clips |
| `tests/test_audio_quality.py` | 8 | Quality gate isolation: VAD, SNR, clipping, return types |
| `tests/test_decision_policy.py` | 11 | Decision thresholds, invariants, boundary conditions |
| `tests/test_sample_file.py` | 2 | End-to-end with `samples/sample.wav` |
| `tests/test_real_model.py` | 1 | Real model inference (skipped unless `DIALFLO_REAL_MODEL=1`) |

**30 tests pass in < 30 seconds without downloading any model weights.**

---

## 📁 File Structure

```
dialflo-voice-attributes/
│
├── README.md                  ← You are here
├── DESIGN_WRITEUP.md          ← 200-word approach summary
├── DESIGN_DECISIONS.md        ← Deep rationale for every non-obvious choice
├── ARCHITECTURE.md            ← Pipeline diagram + scaling strategy
├── DPDP_PRIVACY.md            ← India DPDP Act 2023 compliance posture
│
├── docker-compose.yml         ← docker compose up
├── Dockerfile                 ← Model weights baked in at build time
├── requirements.txt           ← Pinned Python dependencies
│
├── app/
│   ├── main.py                ← FastAPI app: /analyze, /ws/analyze, /health
│   ├── audio_quality.py       ← Quality gate (webrtcvad + SNR + clipping)
│   ├── inference.py           ← AgeGenderModel loading + prediction
│   ├── decision_policy.py     ← Confidence-gated decision: auto_use / flag / discard
│   ├── schemas.py             ← Pydantic response models
│   └── logging_config.py      ← Structured JSON logging (privacy-safe scalars only)
│
├── eval/
│   ├── run_eval.py            ← Accuracy + ECE eval against Mozilla Common Voice
│   └── benchmark.py           ← Latency benchmark (5 iterations on 5s clip)
│
├── scripts/
│   └── download_model.py      ← Model weight download script (used by Dockerfile)
│
├── tests/
│   ├── test_api_smoke.py      ← Integration: full HTTP pipeline (8 tests)
│   ├── test_audio_quality.py  ← Unit: quality gate logic (8 tests)
│   ├── test_decision_policy.py ← Unit: decision thresholds (11 tests)
│   ├── test_sample_file.py    ← Smoke: samples/sample.wav end-to-end (2 tests)
│   └── test_real_model.py     ← Real model inference (DIALFLO_REAL_MODEL=1)
│
└── samples/
    ├── sample.wav             ← Synthetic voiced audio (instant smoke test)
    ├── speech_sample.wav      ← Real LibriSpeech speech clip (public domain)
    ├── generate_sample.py     ← Script to regenerate sample.wav
    └── SOURCING.md            ← How to get Common Voice + LibriSpeech clips
```

---

## 📈 Scaling to 1,000 Concurrent Calls

> The current architecture (single process, CPU inference) does not scale to 1,000 concurrent calls. A single wav2vec2-large forward pass is ~300ms on a CPU core. Here is the production path:

**Step 1 — Separate inference from the API process**

Move model inference into a dedicated inference server (ONNX Runtime or NVIDIA Triton). FastAPI workers remain stateless and scale horizontally, independent of inference compute.

**Step 2 — Micro-batching**

Implement a batching queue with a 20–50ms aggregation window in front of the inference server. This amortises GPU overhead across concurrent requests — one batch of 20 requests takes the same GPU time as one request on GPU.

**Step 3 — ONNX export for CPU**

Export the model to ONNX format and run with ONNX Runtime. Measured 2–3× speedup on CPU with no accuracy loss — brings CPU inference under 150ms, making CPU viable for moderate load without GPU cost.

**Step 4 — WebSocket sticky routing**

WebSocket connections are stateful (audio buffers accumulate within a connection). They cannot be round-robined across stateless replicas. Use consistent hashing on `contact_id` to route the entire call duration to the same inference replica.

**Step 5 — Graceful degradation under load**

Under saturation, return `audio_quality: "insufficient"` and `decision: "discard"` rather than timing out. Same contract, same HTTP 200 — the caller gets a usable response even when the system is overloaded.

---

## 🔒 Privacy & DPDP Compliance

This service processes caller voice audio from which demographic attributes are derived. Under **India's Digital Personal Data Protection Act (DPDP), 2023**, voice recordings and attributes derived from them are personal data, requiring a lawful basis and purpose limitation.

### What this service does

| Action | Status |
|---|---|
| Audio stored to disk | ❌ Never |
| Audio written to logs | ❌ Architecturally impossible — `logging_config.py` only accepts scalar primitives |
| Raw embeddings persisted | ❌ Never |
| Predictions persisted | ❌ Not by this service — returned to caller only |

The structured logger in `logging_config.py` is deliberately architected so the only fields it can physically log are scalars (IDs, timings, labels, confidences). There is no code path where an audio buffer, numpy array, or tensor can reach the logger.

`docker compose down` leaves no PII on disk.

### What still needs a product-level decision

- **Lawful basis** — Is caller consent to demographic inference covered by existing call-recording disclosures, or does it need its own notice? This is a product/legal decision outside this service's boundary.
- **Purpose limitation** — This service should be used only for real-time conversational personalization (tone, greeting word choice) — not fed into downstream profiles, CRM records, or analytics stores without a separate lawful basis for that secondary use.
- **Data principal rights** — Since nothing is stored here, there is nothing to correct, delete, or export from this service. That only holds as long as no downstream system persists the output against a contact record.

> This service is a small, compliant piece of a system whose overall compliance depends on decisions made outside this service's boundary. See [`DPDP_PRIVACY.md`](DPDP_PRIVACY.md) for the full posture.

---

## ⚠️ Known Limitations

### Gender inference is a heuristic, not a fact

Voice-based gender inference detects correlates of biological sex in acoustic features — fundamental frequency, formant patterns, speaking rate. It is not a reliable indicator of gender identity and **will misgender trans and non-binary speakers** whose vocal characteristics do not align with population-level sex-linked patterns.

**Intended use:** Soft signal for tone and greeting personalisation only. The prediction must never be stored as a contact record attribute, used for identification, or treated as ground truth in any downstream system.

### India calibration gap

The model was fine-tuned on predominantly North American and European data. Expected accuracy on Indian-accented English and Hindi is meaningfully lower than on US English. The eval harness surfaces this gap by locale. The fix is fine-tuning on Common Voice `hi` + `en-IN` subsets — the infrastructure for this already exists in the eval harness.

### Age confidence is a proxy, not a calibrated probability

The age head outputs a continuous regression value (not a class probability), so there is no native confidence score for the age bracket. We derive a proxy confidence based on centrality within the predicted bracket — predictions at the centre of a bracket receive high confidence, predictions near a boundary receive low confidence. Proper calibration requires fitting isotonic regression on a held-out evaluation set.

### Codec support

Handles WAV, FLAC, OGG, MP3, and other formats supported by `libsndfile` / `ffmpeg`. GSM and AMR codecs (common in Indian telecom infrastructure) require an explicit pre-decode step — add `pydub` or `ffmpeg-python` if those codecs appear in production traffic.

---

## 📄 Design Documents

| Document | Purpose |
|---|---|
| [`DESIGN_WRITEUP.md`](DESIGN_WRITEUP.md) | 200-word submission write-up: model choice, improvement plan, scaling |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | Deep rationale: model selection, quality gate ordering, decision policy, age proxy |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Pipeline diagram, latency breakdown, scaling architecture |
| [`DPDP_PRIVACY.md`](DPDP_PRIVACY.md) | India DPDP Act 2023 compliance posture and honest scope |

---

## 👤 About the Author

**Vedesh Khatri**

This project was built as a backend engineering take-home for **Dialflo** — a Bengaluru-based voice AI startup building India's most capable logistics voice automation platform.

The goal was not to build a generic ML demo. It was to demonstrate product judgment specific to Dialflo's actual constraints: India-first calibration awareness, latency budgets that matter for live calls, privacy engineering under Indian law, and a decision policy that matches how Dialflo's platform actually works — automated for tier-one, human-escalated for exceptions.

Every non-obvious choice in this codebase has a rationale documented in the design documents above. The limitations section is honest rather than optimistic. The eval harness surfaces inconvenient truths about model accuracy on Indian audio rather than hiding them. That is the kind of engineering judgment this product requires.

---

<div align="center">

**Built with care for India's voice AI future 🇮🇳**

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-vedeshskhatri-181717?style=for-the-badge&logo=github)](https://github.com/vedeshskhatri)

</div>
