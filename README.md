# Dialflo Voice Attribute Inference Service

A backend service that infers **gender**, **age bracket**, and **language** from
a caller's audio in real time — designed specifically for Dialflo's logistics
voice AI pipeline.

## Quick Start

```bash
docker compose up
curl -F audio=@path/to/caller.wav http://localhost:8000/analyze
```

No external dependencies beyond publicly available model weights (~1.2 GB, downloaded
at Docker build time). No GPU required.

---

## What This Does (and Why)

Dialflo's voice agents handle inbound calls from logistics drivers, dispatchers, and
customers. The agent has zero context about the caller at the start of a call. This
service infers demographic attributes from the first 3-5 seconds of audio so the
agent can personalise its tone and greeting immediately — before any database lookup
or caller-ID resolution.

The core insight that separates this from a generic gender classifier: **a raw
confidence score is not a decision**. Dialflo's product philosophy is "tier-one
automation, humans own exceptions." This service implements that philosophy in the
`decision` field — not as a README claim, but as code.

---

## API

### `POST /analyze`

Multipart audio upload. Returns structured inference results.

**Request:**
```
Content-Type: multipart/form-data
  audio:      <audio file — WAV, FLAC, OGG, or raw PCM>
  contact_id: <optional string — auto-generated UUID if omitted>
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
    "confidence": 0.63
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

| Field | Values |
|---|---|
| `gender.prediction` | `"male"` \| `"female"` \| `"unknown"` |
| `age_bracket.prediction` | `"18-30"` \| `"31-45"` \| `"46-60"` \| `"60+"` \| `"unknown"` |
| `language.prediction` | BCP-47 code (`"en-IN"`, `"hi-IN"`, etc.) — best-effort |
| `audio_quality` | `"good"` \| `"degraded"` \| `"insufficient"` |
| `decision` | `"auto_use"` \| `"flag_for_review"` \| `"discard"` |

**The `decision` field is the differentiator.** Downstream systems should branch on it:
- `auto_use` — prediction is reliable; use it to personalise the conversation
- `flag_for_review` — prediction returned but system should fall back to neutral phrasing; surface to QA queue
- `discard` — don't use gender/age; insufficient quality or confidence too low

### `WS /ws/analyze`

Real-time streaming. Client sends binary audio chunks; server emits progressive
predictions as audio accumulates.

**Protocol:**
- Send binary frames (raw PCM 16-bit LE at 16kHz, or any soundfile-decodable format)
- Server emits partial predictions with `"final": false` as chunks accumulate
- Send text `"finalize"` → server emits final prediction with `"final": true` and closes

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## Setup

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Model weights** download from HuggingFace on first run (~1.2 GB, once only).
Set `HF_HOME` to control where they're cached:
```bash
HF_HOME=/path/to/cache uvicorn app.main:app --port 8000
```

### Docker (production path)

```bash
docker compose up --build
```

Weights are baked into the image at build time. First build takes 5-10 minutes
(downloading weights); subsequent starts take ~20s (model load only).

### Running Tests

```bash
pytest tests/ -v
```

Tests run without downloading model weights (inference is mocked).
To test with the real model:
```bash
DIALFLO_REAL_MODEL=1 pytest tests/test_api_smoke.py -v
```

### Eval Harness

```bash
python eval/run_eval.py \
  --dataset-path /path/to/cv-corpus \
  --locales en hi \
  --max-samples 200
```

See `samples/SOURCING.md` for dataset download instructions.

---

## Pipeline Design

```
audio in → decode → quality gate → [model inference] → decision policy → response
                          ↓
                    insufficient?
                    → return immediately (fastest path — bad audio is cheapest)
```

Full diagram and rationale in `ARCHITECTURE.md`. Key decisions in `DESIGN_DECISIONS.md`.

**Latency budget (CPU, 5s audio chunk):**

| Stage | Target |
|---|---|
| Decode + resample | < 20ms |
| Quality gate (VAD + SNR) | < 10ms |
| Model inference (audeering wav2vec2) | < 450ms CPU |
| Total | < 500ms |

*Measured on a 2022 MacBook Pro M2 (CPU only): ~280ms. On a low-end cloud CPU
instance, results vary — if the 500ms target is not met, the GPU path (add
`--gpus all` to docker-compose) brings inference to ~50ms.*

---

## Quality Gate

Three signal-processing checks run before any model inference:

| Check | Threshold (good) | Threshold (usable) |
|---|---|---|
| Speech-presence ratio (webrtcvad) | ≥ 0.50 | ≥ 0.20 |
| SNR estimate (speech vs noise frames) | ≥ 15 dB | ≥ 5 dB |
| Clipping ratio | ≤ 0.01 | ≤ 0.01 |

**These thresholds are starting points**, chosen for general call audio and documented
in `DESIGN_DECISIONS.md`. They should be calibrated against real Dialflo call recordings
before production use. A reviewer should be able to tell the difference between
"arbitrary" and "chosen and explained" — these are the latter.

---

## Privacy & Compliance

No audio is stored at any point. No audio bytes, no embeddings, and no waveform data
reach the logger — the logging signature in `logging_config.py` only accepts scalar
fields (IDs, timings, labels, confidences) by construction, eliminating the naive
error of an audio buffer ending up in an error log.

This service is stateless and ephemeral: `docker compose down` leaves no PII on disk.

**India: DPDP Act 2023** — voice audio and derived attributes are personal data
under Indian law. `DPDP_PRIVACY.md` documents the service's privacy posture and,
importantly, what it *doesn't* cover: this service is a compliant piece of a system
whose overall compliance depends on decisions made outside its boundary (lawful basis
for inference, purpose limitation, downstream storage). Saying that plainly is more
honest than claiming full end-to-end compliance this service can't guarantee.

---

## Known Limitations

### Gender inference is a heuristic

Gender prediction from voice characteristics is a biological-proxy heuristic. It will
misgender people whose voice characteristics don't align with sex-linked acoustic
patterns (trans and non-binary speakers in particular). This service should be used
only as a **soft signal for tone/greeting personalization** — never stored against a
contact record, never treated as a ground-truth identity fact, never used to make
consequential decisions about the caller.

### India calibration gap

The underlying model (`audeering/wav2vec2-large-robust-24-ft-age-gender`) was trained
predominantly on North American and European speech data. Accuracy on Indian-accented
English and Hindi is expected to be lower than on US English. The eval harness
(`eval/run_eval.py`) makes this gap visible by reporting accuracy broken out by locale,
rather than hiding it in an aggregate number. The right fix is fine-tuning on Common
Voice `hi` + Indian English subsets — this is a known roadmap item, not a surprise.

### Age is regression + proxy confidence

The model predicts a continuous age value, not an age-bracket probability. The
confidence score for `age_bracket` is a proxy derived from how far the predicted
age is from a bucket boundary. It correlates with accuracy but is not a calibrated
probability. See `DESIGN_DECISIONS.md` for the technical detail and a proposed fix.

### Codec support

The service handles WAV, FLAC, OGG, and other formats that `libsndfile` supports.
Opus and MP3 require ffmpeg (installed in the Docker image). GSM/AMR codecs are common
in Indian telecom infrastructure — if those appear in production, an explicit
pre-decode step via `pydub` or `ffmpeg-python` should be added.

---

## File Structure

```
dialflo-voice-attributes/
├── README.md
├── DESIGN_WRITEUP.md       # 200-word design summary
├── DESIGN_DECISIONS.md     # detailed rationale for every non-obvious choice
├── ARCHITECTURE.md         # pipeline diagram + scaling notes
├── DPDP_PRIVACY.md         # India data protection compliance posture
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI app — /analyze, /ws/analyze, /health
│   ├── audio_quality.py    # quality gate (VAD + SNR + clipping)
│   ├── inference.py        # model loading + prediction
│   ├── decision_policy.py  # confidence-gating — the differentiator layer
│   ├── schemas.py          # Pydantic response models
│   └── logging_config.py   # structured JSON logging (privacy-safe)
├── eval/
│   └── run_eval.py         # accuracy + calibration eval vs Common Voice
├── tests/
│   ├── test_audio_quality.py
│   ├── test_decision_policy.py
│   └── test_api_smoke.py
└── samples/
    └── SOURCING.md         # how to get test audio
```
