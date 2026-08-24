"""
Structured JSON logging. Deliberately narrow: the log call signature only
accepts scalar fields, so there is no code path where an audio buffer or
embedding tensor can end up in a log line. See DPDP_PRIVACY.md.
"""
import json
import logging
import sys
import time
from typing import Optional

logger = logging.getLogger("dialflo.voice_attributes")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(_handler)


def log_request(
    request_id: str,
    stage_timings_ms: dict,
    audio_quality: str,
    decision: str,
    gender_pred: Optional[str] = None,
    gender_conf: Optional[float] = None,
    age_pred: Optional[str] = None,
    age_conf: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    record = {
        "ts": time.time(),
        "request_id": request_id,
        "stage_timings_ms": stage_timings_ms,
        "audio_quality": audio_quality,
        "decision": decision,
        "gender_pred": gender_pred,
        "gender_conf": gender_conf,
        "age_pred": age_pred,
        "age_conf": age_conf,
        "error": error,
    }
    logger.info(json.dumps(record))
