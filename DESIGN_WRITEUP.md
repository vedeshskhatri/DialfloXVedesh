# Design Write-Up

## Approach

I chose `audeering/wav2vec2-large-robust-24-ft-age-gender` as the primary model:
a wav2vec2 backbone fine-tuned on natural conversational speech (MSP-Podcast) that
outputs age and gender directly from raw waveform. The key advantage over building
a custom head on top of a generic speaker embedding is that it eliminates a training
step and produces outputs calibrated on actual phone-style speech, not acted or
read audio.

The architectural decision I'm most confident in is running the audio quality gate
*before* the model, not after. Logistics calls have background noise from trucks,
warehouses, and compressed telecom codecs. Bad audio shouldn't burn the full
inference budget — it should fail fast. A `webrtcvad` + SNR + clipping check takes
under 10ms and short-circuits the expensive model call entirely.

The `decision` field (`auto_use / flag_for_review / discard`) encodes Dialflo's own
product philosophy — "humans own exceptions" — as code rather than leaving it as
an implicit convention for API consumers to implement inconsistently.

## How I Would Improve It

Fine-tune the age/gender heads on Mozilla Common Voice Hindi and Indian English
subsets. The current model has a known accuracy gap on Indian-accented audio —
the eval harness surfaces this explicitly rather than hiding it.

Export the model to ONNX for ~2-3x CPU speedup, likely bringing inference under
200ms without a GPU requirement.

## Scaling to 1,000 Concurrent Calls

Move model inference out of the FastAPI process into a dedicated ONNX Runtime
or Triton inference server. Add a micro-batch queue (20-50ms window) in front of
it so GPU utilisation amortises across concurrent requests instead of one
forward-pass per request. Scale the API layer horizontally (it's stateless);
keep inference replicas behind a queue with backpressure. For WebSocket calls,
use sticky routing so buffer state stays on the same replica for the full call duration.
