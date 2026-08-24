"""
FastAPI application — Dialflo Voice Attribute Inference Service.

Endpoints:
  GET  /health            → liveness probe
  POST /analyze           → multipart audio upload, returns AnalyzeResponse
  WS   /ws/analyze        → streaming audio, emits progressive predictions

Design principle: no audio bytes survive beyond this module's request scope.
See DPDP_PRIVACY.md for the full privacy posture documentation.
"""
import io
import time
import uuid

import numpy as np
import soundfile as sf
import os
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


from app import audio_quality, decision_policy, inference
from app.logging_config import log_request
from app.schemas import (
    AgeBracketEstimate,
    AnalyzeResponse,
    GenderEstimate,
    LanguageEstimate,
)

app = FastAPI(
    title="Dialflo Voice Attribute Inference",
    description=(
        "Infers gender, age bracket, and language from a caller's audio. "
        "Designed for Dialflo's logistics voice AI pipeline. "
        "No audio is stored beyond the duration of a request."
    ),
    version="0.1.0",
)

# Mount samples directory for demo test audio
if os.path.exists("samples"):
    app.mount("/samples", StaticFiles(directory="samples"), name="samples")


TARGET_SR = 16_000          # Hz — all inference runs at 16kHz mono
MIN_USABLE_SECONDS = 1.0    # shorter clips get 'insufficient' immediately
WS_MIN_BUFFER_SECONDS = 1.5 # WebSocket: buffer before emitting partial result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_to_mono16k(raw_bytes: bytes) -> np.ndarray:
    """
    Decode audio bytes (any format soundfile handles: WAV, FLAC, OGG, etc.)
    to 16kHz mono float32 PCM.

    Raises ValueError on unreadable audio. Caller is responsible for turning
    that into an 'insufficient' response rather than a 500.

    Note: ffmpeg is not called here — soundfile handles most logistics-world
    codecs (GSM-WAV, mu-law WAV, FLAC). For Opus/MP3 add a pre-decode step
    using pydub or ffmpeg-python if needed in production.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        raise ValueError(f"Could not decode audio: {e}") from e

    if data.ndim > 1:
        data = data.mean(axis=1)   # downmix multichannel to mono

    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    return data


def _run_pipeline(raw_bytes: bytes, contact_id: str) -> AnalyzeResponse:
    """
    Full inference pipeline: decode → quality gate → inference → decision.

    This single function is shared by both the REST and WebSocket handlers so
    the inference logic is never forked. See ARCHITECTURE.md for the pipeline
    diagram.
    """
    t0 = time.monotonic()
    timings: dict = {}
    request_id = str(uuid.uuid4())

    # --- Stage 1: Decode ---
    try:
        samples = _decode_to_mono16k(raw_bytes)
    except ValueError as e:
        log_request(request_id, {"decode_ms": None}, "insufficient", "discard",
                    error=f"decode_failed:{type(e).__name__}")
        return _insufficient_response(contact_id, 0)

    timings["decode_ms"] = int((time.monotonic() - t0) * 1000)

    if len(samples) / TARGET_SR < MIN_USABLE_SECONDS:
        total_ms = int((time.monotonic() - t0) * 1000)
        log_request(request_id, {**timings, "total_ms": total_ms},
                    "insufficient", "discard", error="clip_too_short")
        return _insufficient_response(contact_id, total_ms)

    # --- Stage 2: Quality gate (cheap — runs before any model) ---
    t1 = time.monotonic()
    quality = audio_quality.assess(samples, TARGET_SR)
    timings["quality_gate_ms"] = int((time.monotonic() - t1) * 1000)

    if quality.quality == "insufficient":
        total_ms = int((time.monotonic() - t0) * 1000)
        log_request(request_id, {**timings, "total_ms": total_ms},
                    quality.quality, "discard")
        return _insufficient_response(contact_id, total_ms)

    # --- Stage 3: Model inference ---
    t2 = time.monotonic()
    raw_pred = inference.predict(samples, TARGET_SR)
    timings["inference_ms"] = int((time.monotonic() - t2) * 1000)

    # --- Stage 4: Decision policy ---
    min_conf = min(raw_pred.gender_confidence, raw_pred.age_confidence)
    dec = decision_policy.decide(min_conf, quality)

    total_ms = int((time.monotonic() - t0) * 1000)
    timings["total_ms"] = total_ms

    log_request(
        request_id, timings, quality.quality, dec.decision,
        gender_pred=raw_pred.gender_label,
        gender_conf=raw_pred.gender_confidence,
        age_pred=raw_pred.age_bracket,
        age_conf=raw_pred.age_confidence,
    )

    # On discard — return unknown predictions but still surface quality/decision
    gender_out = raw_pred.gender_label if dec.decision != "discard" else "unknown"
    age_out    = raw_pred.age_bracket  if dec.decision != "discard" else "unknown"

    lang_out = None
    if raw_pred.language_label is not None:
        lang_out = LanguageEstimate(
            prediction=raw_pred.language_label,
            confidence=raw_pred.language_confidence,
        )

    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderEstimate(prediction=gender_out, confidence=raw_pred.gender_confidence),
        age_bracket=AgeBracketEstimate(prediction=age_out, confidence=raw_pred.age_confidence),
        language=lang_out,
        processing_ms=total_ms,
        audio_quality=quality.quality,
        decision=dec.decision,
    )


def _insufficient_response(contact_id: str, elapsed_ms: int) -> AnalyzeResponse:
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderEstimate(prediction="unknown", confidence=0.0),
        age_bracket=AgeBracketEstimate(prediction="unknown", confidence=0.0),
        language=None,
        processing_ms=elapsed_ms,
        audio_quality="insufficient",
        decision="discard",
    )


# ---------------------------------------------------------------------------
# Web UI & REST endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["ui"])
@app.get("/demo", response_class=HTMLResponse, tags=["ui"])
def demo_ui():
    """Interactive web demo for voice attribute inference."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Dialflo Voice Attribute Inference API is running.</h2>")


@app.get("/health", tags=["ops"])
def health():

    """Liveness probe. Returns 200 if the service is running."""
    return {"status": "ok", "service": "dialflo-voice-attributes"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
async def analyze(
    audio: UploadFile = File(..., description="Audio file (WAV, FLAC, OGG, or raw PCM)"),
    contact_id: str = Form(default=None, description="Optional caller ID; auto-generated if omitted"),
):
    """
    Infer gender, age bracket, and language from an uploaded audio file.

    Returns a structured response with confidence scores and a decision field
    (auto_use / flag_for_review / discard) indicating whether the prediction
    is reliable enough to act on.
    """
    raw_bytes = await audio.read()
    cid = contact_id or str(uuid.uuid4())
    try:
        return _run_pipeline(raw_bytes, cid)
    except Exception:
        # Catch-all: never leak a traceback or audio bytes in the response.
        return JSONResponse(
            status_code=200,
            content=_insufficient_response(cid, 0).model_dump(),
        )


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket):
    """
    Real-time streaming inference.

    Protocol:
      - Client sends binary frames (raw PCM 16-bit little-endian at 16kHz, or
        any format soundfile can decode if sent as a complete chunk).
      - Server buffers chunks until MIN_BUFFER_SECONDS of audio has accumulated,
        then emits a partial prediction JSON with "final": false.
      - Client sends text "finalize" → server emits final prediction with
        "final": true and closes.
      - On disconnect, server closes silently.

    Same _run_pipeline() call as the REST endpoint — no forked logic.
    """
    await websocket.accept()
    buffer = bytearray()
    contact_id = str(uuid.uuid4())
    # 16-bit PCM: 2 bytes per sample, 16000 samples/sec
    min_bytes = int(TARGET_SR * 2 * WS_MIN_BUFFER_SECONDS)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                buffer.extend(message["bytes"])
                if len(buffer) >= min_bytes:
                    result = _run_pipeline(bytes(buffer), contact_id)
                    await websocket.send_json({**result.model_dump(), "final": False})

            elif "text" in message and message.get("text") == "finalize":
                result = _run_pipeline(bytes(buffer), contact_id)
                await websocket.send_json({**result.model_dump(), "final": True})
                break

    except WebSocketDisconnect:
        pass   # client disconnected cleanly — no error to surface
