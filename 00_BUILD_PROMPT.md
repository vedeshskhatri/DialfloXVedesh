# Build Prompt — Dialflo Backend Engineering Assignment (Extended)

Paste this whole file into Antigravity as the task prompt. It contains full context —
you should not need to ask clarifying questions before starting; where you must make a
judgment call, follow the defaults stated here and note the choice in the README.

## 1. Who this is for

Dialflo (dialflo.ai) — Bengaluru-based seed-stage voice AI startup. They sell AI voice
agents that handle high-volume calls for logistics, recruitment, and D2C companies:
COD confirmations, delivery coordination, appointment reminders, driver/dispatcher
check calls, TMS-integrated status updates. Two things they repeat constantly in their
own marketing and should be treated as ground truth about what they value:

1. **Multilingual, India-first is their actual product wedge**, not a side feature.
2. **"Tier-one automation, humans own exceptions."** Their pitch is that routine calls
   get automated and anything uncertain gets escalated instantly. Trust in the product
   depends on the system knowing when it doesn't know.

This is a job-application take-home. The goal is not just a working service — it's to
demonstrate product judgment specific to this company, not a generic ML demo.

## 2. The literal assignment (must satisfy in full)

Build a backend service that accepts audio (streamed or chunked, HTTP or WebSocket)
and returns estimated **gender** and **age bracket** for the speaker, with confidence
scores, an `audio_quality` flag, and `processing_ms`. Target <500ms end-to-end on a
5-second chunk. No audio persisted beyond the request. Must run via
`docker compose up` with no external deps beyond public model weights. Include a
README (setup, design decisions, model rationale, known limitations), a 200-word
design write-up, at least one test, and a sample audio file or sourcing instructions.

Exact response contract:

```json
{
  "contact_id": "uuid",
  "gender": { "prediction": "male" | "female" | "unknown", "confidence": 0.87 },
  "age_bracket": { "prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown", "confidence": 0.63 },
  "processing_ms": 142,
  "audio_quality": "good" | "degraded" | "insufficient"
}
```

Bonus (do all three — see sections 5-7): WebSocket streaming with progressive
predictions, language/accent field, eval harness against a public dataset with
accuracy + calibration metrics.

## 3. The five things that make this submission different

Do the base assignment properly, then layer these five in. Each one should be visible
in the code, not just claimed in the README.

1. **Confidence-gated decision policy**, not a bare number. See `decision_policy.py`.
2. **India-accent/language calibration awareness** in the eval harness — don't just
   report one aggregate accuracy number; break it out by accent/language subgroup.
3. **DPDP Act 2023 compliance section** in the README — see `DPDP_PRIVACY.md`.
4. **Real observability** — structured logs with a latency breakdown per stage, not a
   single `processing_ms` and a print statement.
5. **Honest limitations section** stating plainly that gender-from-voice is a
   biological-proxy heuristic that will misgender people, and should be positioned as
   a soft personalization signal only — never stored, never treated as identity fact.

Full rationale for each is in `DESIGN_DECISIONS.md` and `DPDP_PRIVACY.md` — read those
before writing the README so the language is consistent.

## 4. Tech stack (defaults — deviate only with a documented reason)

- **API**: FastAPI (async), Uvicorn.
- **Audio quality / VAD**: `librosa` + `webrtcvad` — SNR estimate, silence ratio,
  clipping detection, speech-presence ratio. Pure signal processing, no model needed,
  fast enough to run before the expensive inference step (fail fast on bad audio).
- **Embeddings**: SpeechBrain's ECAPA-TDNN speaker embedding model (pretrained,
  `speechbrain/spkrec-ecapa-voxceleb`) as the feature extractor — it's a strong,
  widely-used, well-documented speaker-characteristic embedding, not something built
  from scratch. Age/gender are estimated with a lightweight classifier head trained or
  fine-tuned on top of that embedding (or use an existing age/gender head if one is
  available — e.g. `audeering/wav2vec2-large-robust-24-ft-age-gender` on Hugging Face,
  which directly outputs age + gender and saves training time — prefer this as the
  primary path, with the SpeechBrain embedding pipeline as a documented alternative).
- **Containerization**: Docker + docker-compose, single service, CPU-only inference
  target (state clearly that GPU is an optimization, not a requirement, since the
  spec asks for something that runs with no external dependencies beyond public
  weights — don't assume the grader has a GPU).
- **Tests**: `pytest`.
- **Eval dataset**: Mozilla Common Voice (has age/gender/accent metadata already —
  this is why it's the right choice over VoxCeleb, which lacks reliable age labels).

## 5. Real-time streaming (bonus)

WebSocket endpoint `/ws/analyze` — client streams raw PCM or Opus chunks; server runs
VAD to detect enough speech has accumulated (~1.5-2s minimum), then emits progressive
predictions with widening confidence as more audio arrives, final prediction on stream
close or a `{"action": "finalize"}` message from client. Reuse the same
`decision_policy.py` logic for gating — don't fork the logic between REST and WS paths.

## 6. Language / accent field

Best-effort field in the response: `"language": {"prediction": "en-IN", "confidence": 0.7}`
using a lightweight language-ID model (e.g. `speechbrain/lang-id-voxlingua107-ecapa`) —
report it as best-effort, not authoritative, same posture as gender/age.

## 7. Eval harness

Script that pulls a Common Voice subset (English + Hindi if feasible), runs inference,
prints: overall accuracy per attribute, accuracy broken out by accent/locale subgroup
where metadata allows, and confidence calibration (reliability diagram data or ECE —
expected calibration error — printed as a table is fine, a plot is a bonus not
required). This is where the "India calibration" differentiator gets its evidence —
the eval harness should make the accuracy gap on non-US-English accents visible, not
hide it.

## 8. File structure to produce

```
dialflo-voice-attributes/
├── README.md
├── DESIGN_WRITEUP.md          # the 200-word piece, standalone
├── DPDP_PRIVACY.md
├── ARCHITECTURE.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py                # FastAPI app, /analyze, /ws/analyze, /health
│   ├── audio_quality.py
│   ├── inference.py
│   ├── decision_policy.py
│   ├── schemas.py             # pydantic models matching the response contract
│   └── logging_config.py
├── eval/
│   └── run_eval.py
├── tests/
│   ├── test_audio_quality.py
│   ├── test_decision_policy.py
│   └── test_api_smoke.py
└── samples/
    └── SOURCING.md            # where to get a legal sample audio file for smoke test
```

## 9. What "done" looks like

`docker compose up` starts the service. `curl -F audio=@samples/sample.wav
http://localhost:8000/analyze` returns a contract-matching JSON in under 500ms on
CPU for a 5s clip (state actual measured latency in the README, don't just claim the
target was hit — if it's not hit on CPU, say so and explain the GPU path). Tests pass.
Eval harness runs against a small Common Voice subset and prints the subgroup
breakdown. README reads like it was written by someone who read Dialflo's product,
not someone who read the assignment PDF in isolation.
