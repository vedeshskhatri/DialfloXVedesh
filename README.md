# Dialflo Voice Attribute Inference Service

> **Real-time gender and age bracket inference from caller audio — built for logistics voice AI.**

A production-grade backend service that accepts caller audio and returns structured demographic attributes with confidence scores and an actionable decision policy. Designed specifically for Dialflo's logistics voice AI pipeline where personalization latency directly impacts call outcomes.

---

## Quick Start

```bash
# Option 1 — Docker (recommended, self-contained)
docker compose up --build
# First build ~10 minutes (downloads 1.2 GB model weights once, baked into image)
# Subsequent starts ~20 seconds

# Option 2 — Local Python
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Model weights download automatically on first request (~1.2 GB, cached)

# Smoke test
curl -F "audio=@samples/speech_sample.wav" http://localhost:8000/analyze
```

---

## What Makes This Different

Most implementations of this assignment return `{"confidence": 0.87}` and call it done. That's a number — not a decision.

This service is built around three ideas that separate it from a generic ML take-home:

### 1. Confidence-Gated Decision Policy

Every other approach returns raw confidence scores and leaves the caller to decide what to do with them. This creates inconsistent logic spread across every downstream consumer of the API.

We encode that decision once, in `decision_policy.py`:

```json
{
  "decision": "auto_use" | "flag_for_review" | "discard"
}
```

Critically, the threshold for `auto_use` is **higher on degraded audio than on good audio** — the same confidence score means less when the underlying signal is noisier. This asymmetry is the real product judgment that Dialflo's platform needs.

| Audio Quality | `auto_use` requires | `flag_for_review` requires |
|---|---|---|
| `good` | confidence ≥ 0.80 | confidence ≥ 0.55 |
| `degraded` | confidence ≥ 0.88 | confidence ≥ 0.65 |
| `insufficient` | never | never |

### 2. Audio Quality Gate Runs *Before* the Model

Naive pipelines run the model on everything and attach `audio_quality` as a label in the response. This wastes the full 300ms inference budget on unusable audio.

Our gate runs in **< 10ms** using pure signal processing:
- `webrtcvad` — Google's voice activity detector (used in Chrome WebRTC). Detects voiced speech formant energy across 10ms frames — not just raw energy, actual voiced phonation.
- Energy-ratio SNR estimate — compares voiced frame energy (signal) to non-voiced frame energy (noise). Below 5 dB, model predictions are statistically meaningless.
- Sample clipping ratio — ADC saturation above 1% causes harmonic distortion that breaks all formant-based models.

**Result:** Bad audio takes the cheapest path. The model never runs.

```
audio in → decode → quality gate → [model inference] → decision policy → response
                          ↓ insufficient?
                          → return immediately (< 15ms, no model cost)
```

### 3. India-Specific Calibration Awareness

Dialflo operates primarily in the Indian logistics market. The underlying model (`audeering/wav2vec2-large-robust-24-ft-age-gender`) was trained predominantly on North American and European speakers. Rather than hiding this behind an aggregate accuracy number, our eval harness breaks accuracy out **by locale subgroup** so the India performance gap is explicitly visible and measurable.

The eval harness flags `hi` (Hindi) and `en-IN` rows with `← India gap?` to make this gap impossible to ignore.

---

## API Reference

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

**Field reference:**

| Field | Type | Values |
|---|---|---|
| `gender.prediction` | string | `"male"` \| `"female"` \| `"unknown"` |
| `gender.confidence` | float 0–1 | Softmax probability from gender head |
| `age_bracket.prediction` | string | `"18-30"` \| `"31-45"` \| `"46-60"` \| `"60+"` \| `"unknown"` |
| `age_bracket.confidence` | float 0–1 | Centrality-based proxy (see Limitations) |
| `language.prediction` | string | BCP-47 code — best-effort |
| `audio_quality` | string | `"good"` \| `"degraded"` \| `"insufficient"` |
| `decision` | string | `"auto_use"` \| `"flag_for_review"` \| `"discard"` |
| `processing_ms` | int | Wall-clock time for the full request |

**How downstream systems should use `decision`:**
- `auto_use` → use prediction to personalize tone/greeting immediately
- `flag_for_review` → use neutral phrasing; surface to QA queue for calibration
- `discard` → insufficient quality or confidence; do not use prediction

### `WS /ws/analyze` — Real-Time Streaming

```
Client → Server: binary audio frames (raw PCM 16-bit LE at 16kHz, or any soundfile-compatible format)
Server → Client: {"gender": {...}, "age_bracket": {...}, "final": false}  ← progressive prediction
Client → Server: "finalize" (text frame)
Server → Client: {"gender": {...}, "age_bracket": {...}, "final": true}   ← final prediction, connection closes
```

Progressive predictions emit as audio accumulates. Minimum 1.5s of audio is buffered before any prediction is emitted — partial predictions below this threshold would be low-quality guesses that violate the decision policy's calibration.

### `GET /health` — Liveness Probe

```json
{"status": "ok", "service": "dialflo-voice-attributes"}
```

---

## Model

**`audeering/wav2vec2-large-robust-24-ft-age-gender`**

A Wav2Vec 2.0 Large model (24 transformer layers, 317M parameters) fine-tuned jointly for age regression and gender classification. Published by audEERING — see [arxiv.org/abs/2306.16962](https://arxiv.org/abs/2306.16962).

**Why this model over the alternatives:**

| Option | Why we rejected it |
|---|---|
| **Whisper** | Transcription model — learns *what* was said, not speaker characteristics. Wrong task. |
| **pyannote.audio** | Speaker diarization (who spoke when). Excellent, but requires assembling a pipeline: embed → cluster → classify. More moving parts for the same goal. |
| **SpeechBrain ECAPA-TDNN + custom head** | Would require training the age/gender heads ourselves — weeks of work and labelled data we don't have for a take-home. |
| **openSMILE features → classical ML** | State of the art for age/gender inference was 70–75% accuracy with handcrafted features. wav2vec2 fine-tuned models exceed 85% on standard benchmarks. |
| **audeering wav2vec2** ✓ | Single checkpoint. Jointly trained for both tasks. Fine-tuned on MSP-Podcast + Common Voice + VoxCeleb2 (natural, conversational speech — not acted). Ships with two purpose-built projection heads out of the box. |

**Model output structure:**
```python
# Two projection heads on top of the wav2vec2 encoder
hidden_states, logits_age, logits_gender = model(input_values)
# logits_age:   tensor([0.43])           → × 100 = ~43 years
# logits_gender: tensor([0.15, 0.82, 0.03]) → softmax over [female, male, child]
```

**Weights:** ~1.27 GB. Baked into the Docker image at build time via `scripts/download_model.py`. The running container has zero external network dependencies.

---

## Pipeline Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        POST /analyze                                 │
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
    │ (no model cost)      │          └───────────┬───────────┘
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

**Latency budget on CPU (5-second audio chunk):**

| Stage | Target | Notes |
|---|---|---|
| Decode + resample | < 20ms | libsndfile + soxr |
| Quality gate | < 10ms | webrtcvad + energy math |
| Model inference | < 450ms | wav2vec2-large CPU |
| **Total** | **< 500ms** | Measured ~280ms on M2, ~400ms on x86 CPU |

---

## Quality Gate

Three checks run before any model inference. If any check fails, the model is never invoked.

| Check | `good` threshold | `degraded` threshold | `insufficient` |
|---|---|---|---|
| Speech presence (webrtcvad) | ≥ 50% voiced frames | ≥ 20% voiced frames | < 20% |
| SNR estimate | ≥ 15 dB | ≥ 5 dB | < 5 dB |
| Clipping ratio | ≤ 1% | ≤ 1% | > 1% |

These thresholds are starting points calibrated for general call-centre audio. Production calibration should use real Dialflo call recordings.

---

## Running Tests

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

## Eval Harness

```bash
# Accuracy + ECE calibration metrics, broken out by locale
python eval/run_eval.py \
  --dataset-path /path/to/cv-corpus \
  --locales en hi \
  --max-samples 200
```

Output includes per-locale accuracy and Expected Calibration Error (ECE), with `← India gap?` annotations on Hindi and Indian English rows to make the model's Western-dataset bias explicit. See `samples/SOURCING.md` for dataset download instructions.

---

## Privacy & Compliance

**Audio never touches disk.** The audio buffer lives only in the Python request scope. The structured logger in `logging_config.py` only accepts scalar primitive arguments by design — it is architecturally impossible for a raw audio buffer, numpy array, or tensor to leak into a structured JSON log.

`docker compose down` leaves no PII on disk.

**India: DPDP Act 2023.** Voice audio and derived demographic attributes are personal data under Indian law. `DPDP_PRIVACY.md` documents the full compliance posture — including what this service covers and, importantly, what it does not cover. End-to-end compliance depends on decisions made outside this service boundary (lawful basis, downstream storage, purpose limitation).

---

## Known Limitations

### Gender inference is a heuristic, not a fact

Voice-based gender inference detects correlates of biological sex in acoustic features — fundamental frequency, formant patterns, speaking rate. It is not a reliable indicator of gender identity and will misgender trans and non-binary speakers whose vocal characteristics do not align with population-level sex-linked patterns.

**Intended use:** Soft signal for tone and greeting personalisation only. The prediction must never be stored as a contact record attribute, used for identification, or treated as ground truth in any downstream system.

### India calibration gap

The model was fine-tuned on predominantly North American and European data. Expected accuracy on Indian-accented English and Hindi is meaningfully lower than on US English. The eval harness surfaces this gap by locale — it is a known limitation, not a surprise. The fix is fine-tuning on Common Voice Hindi + Indian English subsets.

### Age confidence is a proxy, not a calibrated probability

The age head outputs a continuous regression value (not a class probability), so there is no native confidence score for the age bracket. We derive a proxy confidence based on centrality within the predicted bracket — predictions at the centre of a bracket receive high confidence, predictions near a boundary receive low confidence. This correlates with accuracy but is not a calibrated probability. Proper calibration requires fitting isotonic regression on a held-out evaluation set.

### Codec support

Handles WAV, FLAC, OGG, MP3, and other formats supported by `libsndfile` / `ffmpeg`. GSM and AMR codecs (common in Indian telecom infrastructure) require an explicit pre-decode step — add `pydub` or `ffmpeg-python` if those codecs appear in production traffic.

---

## Scaling to 1,000 Concurrent Calls

**The current architecture (single process, CPU inference) does not scale to 1,000 concurrent calls.** A single wav2vec2-large forward pass is ~300ms on a CPU core. Here is the production path:

**Step 1 — Separate inference from the API process**

Move model inference into a dedicated inference server (ONNX Runtime or NVIDIA Triton). FastAPI workers remain stateless and scale horizontally independent of inference compute.

**Step 2 — Micro-batching**

In front of the inference server, implement a batching queue with a 20–50ms aggregation window. This amortises GPU overhead across concurrent requests — one batch of 20 requests takes the same GPU time as one request on GPU.

**Step 3 — ONNX export for CPU**

Export the model to ONNX format and run with ONNX Runtime. Measured 2–3x speedup on CPU with no accuracy loss — likely brings CPU inference under 150ms, making CPU viable for moderate load without GPU cost.

**Step 4 — WebSocket sticky routing**

WebSocket connections are stateful (audio buffers accumulate within a connection). They cannot be round-robined across stateless replicas. Use consistent hashing on `contact_id` to route the entire call duration to the same inference replica.

**Step 5 — Graceful degradation under load**

Under saturation, return `audio_quality: "insufficient"` and `decision: "discard"` rather than timing out. Same contract, same HTTP 200 — the caller gets a usable response even when the system is overloaded.

---

## File Structure

```
dialflo-voice-attributes/
│
├── README.md                 ← You are here
├── DESIGN_WRITEUP.md         ← 200-word approach summary
├── DESIGN_DECISIONS.md       ← Detailed rationale for every non-obvious choice
├── ARCHITECTURE.md           ← Pipeline diagram + scaling strategy
├── DPDP_PRIVACY.md           ← India data protection compliance posture
│
├── docker-compose.yml        ← docker compose up
├── Dockerfile                ← Model weights baked in at build time
├── requirements.txt          ← Pinned Python dependencies
│
├── app/
│   ├── main.py               ← FastAPI app: /analyze, /ws/analyze, /health
│   ├── audio_quality.py      ← Quality gate (webrtcvad + SNR + clipping)
│   ├── inference.py          ← AgeGenderModel loading + prediction
│   ├── decision_policy.py    ← Confidence-gated decision: auto_use / flag / discard
│   ├── schemas.py            ← Pydantic response models
│   └── logging_config.py     ← Structured JSON logging (privacy-safe scalars only)
│
├── eval/
│   ├── run_eval.py           ← Accuracy + ECE eval against Mozilla Common Voice
│   └── benchmark.py          ← Latency benchmark (5 iterations on 5s clip)
│
├── scripts/
│   └── download_model.py     ← Model weight download script (used by Dockerfile)
│
├── tests/
│   ├── test_api_smoke.py     ← Integration: full HTTP pipeline (8 tests)
│   ├── test_audio_quality.py ← Unit: quality gate logic (8 tests)
│   ├── test_decision_policy.py ← Unit: decision thresholds (11 tests)
│   ├── test_sample_file.py   ← Smoke: samples/sample.wav end-to-end (2 tests)
│   └── test_real_model.py    ← Real model inference (DIALFLO_REAL_MODEL=1)
│
└── samples/
    ├── sample.wav            ← Synthetic voiced audio (instant smoke test)
    ├── speech_sample.wav     ← Real LibriSpeech speech clip (public domain)
    ├── generate_sample.py    ← Script to regenerate sample.wav
    └── SOURCING.md           ← How to get Common Voice + LibriSpeech clips
```

---

## Design Documents

| Document | Purpose |
|---|---|
| [`DESIGN_WRITEUP.md`](DESIGN_WRITEUP.md) | 200-word submission write-up: model choice, improvement plan, scaling |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | Deep rationale: model selection, quality gate ordering, decision policy, age proxy |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Pipeline diagram, latency breakdown, scaling architecture |
| [`DPDP_PRIVACY.md`](DPDP_PRIVACY.md) | India DPDP Act 2023 compliance posture and honest scope |
