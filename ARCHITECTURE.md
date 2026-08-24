# Architecture

## Pipeline

```
audio in (HTTP multipart / WS chunks)
        │
        ▼
[1] Decode + resample (ffmpeg/soundfile → 16kHz mono PCM)
        │
        ▼
[2] Audio quality gate (audio_quality.py)
    - SNR estimate, clipping ratio, speech-presence ratio (VAD)
    - cheap, runs before any model inference — fail fast
    - outputs: "good" | "degraded" | "insufficient"
    - "insufficient" short-circuits: returns unknown/unknown immediately,
      skips inference entirely (saves latency + avoids confident garbage)
        │
        ▼
[3] Feature extraction / embedding (inference.py)
    - speaker/paralinguistic embedding model (see DESIGN_DECISIONS.md)
        │
        ▼
[4] Attribute classification heads
    - gender head, age-bracket head, (bonus) language-ID head
    - each returns prediction + raw confidence
        │
        ▼
[5] Decision policy (decision_policy.py)
    - confidence + audio_quality → auto_use / flag_for_review / discard
    - this is the layer that doesn't exist in a naive implementation
        │
        ▼
[6] Response assembly (schemas.py) + structured log emit (logging_config.py)
        │
        ▼
JSON response matching contract
```

## Why the quality gate runs before inference, not after

A naive pipeline runs the model on everything and only checks quality to annotate the
response. That wastes latency budget on audio that was never going to produce a
trustworthy prediction, and it's the reverse of what Dialflo's own product promise
implies (escalate instead of guessing). Gating first means the "insufficient" path is
also the fastest path — bad audio should be the cheapest case, not the most expensive.

## Streaming path

The WebSocket handler is a thin wrapper around the same five stages — it buffers
incoming chunks until the VAD-detected speech duration crosses a minimum threshold,
runs stages 2-5 on the buffered audio, emits a partial result, and keeps accumulating
until `finalize` or stream close. No separate inference logic for streaming vs batch —
one code path, two entry points. This matters for maintainability and is worth calling
out explicitly in the README as a decision, not an accident.

## Observability

Every request gets a `request_id` (uuid4), and a structured JSON log line with:
`request_id, stage_timings_ms: {decode, quality_gate, embedding, classification,
total}, audio_quality, decision, gender_pred, age_pred, confidence values`. No raw
audio or any audio-derived embedding is included in logs — see DPDP_PRIVACY.md for why
that specific boundary matters.

## Scaling to 1,000 concurrent calls (for the design write-up)

- CPU inference per-request is fine at low volume but won't hold at 1,000 concurrent
  5s-chunk requests; the real lever is **batching** — a short-lived micro-batch queue
  (e.g. 20-50ms window) in front of the model so GPU/CPU utilization amortizes across
  concurrent requests instead of one-request-per-forward-pass.
- Move the model serving out of the FastAPI process into a dedicated inference server
  (Triton Inference Server or a simple ONNX Runtime service) so the API layer stays
  stateless and horizontally scalable independent of GPU capacity.
- Horizontal pod autoscaling on the API layer, fixed pool of GPU inference replicas
  behind a queue (Redis/streaming queue) with backpressure — return `audio_quality:
  "insufficient"`-style graceful degradation under load rather than timing out.
- For the WebSocket path specifically: sticky routing per call (same instance handles
  the whole call) to avoid re-buffering state across replicas.
