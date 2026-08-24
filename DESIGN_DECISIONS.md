# Design Decisions & Rationale

## Model Choice: `audeering/wav2vec2-large-robust-24-ft-age-gender`

**Why not train from scratch?** The assignment is a 2-day take-home. Training an
age/gender model from scratch would take weeks and a labelled corpus we don't have.
The right decision is to use a strong pretrained model and invest the time in the
surrounding system (quality gate, decision policy, compliance), which is where the
real product judgment lives.

**Why audeering over SpeechBrain ECAPA-TDNN + custom head?**
- The audeering model is a single checkpoint that jointly outputs both age and gender
  directly from raw waveform — no separate feature extractor + head training step.
- It's fine-tuned on MSP-Podcast, a natural speech corpus (not acted speech), which
  makes it better suited to real call-centre audio than models trained on read speech.
- It's well documented, widely used, and the HuggingFace hub makes weight distribution
  trivial for the Docker build requirement.

**Known model limitation — India calibration gap:**
The model was trained on predominantly North American/European audio.
Expected accuracy on Indian-accented English and Hindi is lower than on US English.
The eval harness (`eval/run_eval.py`) makes this gap visible by breaking accuracy
out by Common Voice locale, rather than hiding it in an aggregate number.
Proposed fix with more time: fine-tune the age/gender heads on Common Voice
`hi` + `en-IN` subsets, which have age and gender labels.

**Known model limitation — gender binary assumption:**
The model outputs 3 classes (female / male / other) and maps `other` → `unknown`
in our schema. Gender prediction from voice pitch/timbre is a biological-proxy
heuristic — it will misgender people whose voice characteristics don't align with
sex-linked acoustic patterns. This service should be used only for soft
personalization signals (tone, greeting word choice), never stored as an identity
fact, never used rigidly. See the Limitations section in README.md.

---

## Audio Quality Gate: Why It Runs *Before* Inference

The naive pipeline runs the model on everything and uses `audio_quality` only as
a label in the response. That has two problems:

1. **Wasted latency**: a 5-second chunk of warehouse noise takes the full 400ms
   inference budget to produce a prediction nobody should trust.
2. **Wrong escalation signal**: Dialflo's own product promise is "instant escalation
   on uncertainty." An `audio_quality: "insufficient"` response that comes after
   a full model run is slower and more expensive than a response that bails early.

The gate runs fast (< 10ms for a 5s clip) using only signal processing:
- `webrtcvad` for speech-presence detection
- Energy-ratio SNR estimate (speech frames vs noise frames)
- Sample clipping ratio

If the gate returns `insufficient`, the model never runs. Bad audio = cheapest path.

**Threshold values** (`MIN_SPEECH_RATIO_GOOD = 0.5`, `MIN_SNR_DB_GOOD = 15.0`, etc.)
are documented starting points tuned against general call audio. They should be
calibrated against real Dialflo call recordings — the README flags this explicitly
so a reviewer knows they're not arbitrary.

**webrtcvad limitation on synthetic audio:**
webrtcvad uses a multi-band energy model tuned for voiced speech formants. Pure
sine waves and white noise are not detected as speech even at high amplitude.
This is correct behaviour — in production, real caller audio has voiced formants.
The test suite documents this explicitly (`test_sine_wave_is_low_speech_ratio`).

---

## Decision Policy: Why a `decision` Field Exists

Every other implementation of this assignment will return `{"confidence": 0.87}`.
That's a number, not a decision.

Dialflo's public product positioning is: *"tier-one automation, humans own exceptions."*
Their call platform needs to know whether to act on the prediction, queue it for
human review, or ignore it. Returning a raw confidence number pushes that logic into
the caller — and it will be implemented inconsistently by whoever consumes the API.

The `decision` field encodes that logic once, consistently, using quality-aware
thresholds: `auto_use / flag_for_review / discard`. Critically, the threshold for
`auto_use` is higher on `degraded` audio than on `good` audio — the same confidence
score is worth less when the underlying signal is noisy.

**Threshold values** (`AUTO_USE_CONFIDENCE`, `FLAG_CONFIDENCE` in `decision_policy.py`)
are documented starting points. Calibrating them properly requires a held-out labelled
dataset of Dialflo call audio, which this take-home doesn't have. The README says
this plainly rather than implying the values are production-ready.

---

## Age Confidence: Why It's a Proxy, Not a Real Probability

The audeering age head is a regression output — it predicts a continuous age value,
not a class probability. There's no native confidence score. We derive a proxy
confidence based on how far the predicted age is from a bucket boundary (high
confidence when the prediction is solidly inside a bucket, low when it's near the
edge of two brackets). This is explicitly documented in `inference.py` and the
README so a reviewer doesn't mistake it for a calibrated probability.

Proper approach with more time: use the eval harness to measure the correlation
between this proxy and actual bracket accuracy, and if it's poorly calibrated,
replace it with a proper Platt scaling or isotonic regression calibration step.

---

## Streaming (WebSocket): One Code Path, Two Entry Points

The WebSocket handler (`/ws/analyze`) shares the same `_run_pipeline()` function
as the REST handler (`/analyze`). This is a deliberate choice — forked logic
between batch and streaming paths is a maintenance hazard and a bug surface.
The tradeoff is that the streaming path inherits the same minimum-length check:
the service won't emit a partial prediction until at least 1.5 seconds of audio
has been buffered. Progressive predictions below that threshold would be low-quality
guesses that violate the decision policy's calibration.

---

## What Would Change With More Time

1. **Dual-head model**: Load the audeering age head as a separate regression head
   rather than deriving age from gender logits. Audeering publishes the code for this
   — it was skipped here for time, and the approximation is documented.

2. **Fine-tune on Indian audio**: Use Common Voice `hi` + Indian English subsets to
   fine-tune the gender/age heads. The eval harness already measures the accuracy gap
   that motivates this.

3. **Calibrate thresholds**: Measure decision policy thresholds against real call data
   and use isotonic regression to calibrate the age confidence proxy.

4. **Streaming buffer with proper VAD**: Replace the fixed-byte-count buffer with
   a VAD-triggered buffer that emits as soon as enough voiced speech frames have
   accumulated, not just enough raw bytes.

5. **ONNX export**: Export the model to ONNX format for ~2-3x CPU inference speedup
   with ONNX Runtime, which would likely bring latency well inside the 500ms target
   even on CPU without GPU.

---

## Scaling to 1,000 Concurrent Calls

See `ARCHITECTURE.md` §Scaling for the full treatment. Short version:
- CPU inference per request won't hold at 1,000 concurrent 5-second chunks.
- The fix is micro-batching (20-50ms window) in front of a GPU inference server
  (Triton or ONNX Runtime), with the FastAPI layer staying stateless and horizontally
  scalable independent of the inference compute.
- WebSocket calls need sticky routing (same replica handles the whole call) to
  avoid re-buffering state across replicas.
- Graceful degradation under load: return `audio_quality: "insufficient"` and
  `decision: "discard"` rather than timing out — same pattern as the quality gate.
