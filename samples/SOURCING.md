# Sample Audio Sourcing

The test suite (`tests/test_api_smoke.py`) generates synthetic WAV files in memory —
no external audio file is needed to run the tests.

For a real-world smoke test against the running service, use one of the following:

## Option 0: Included Sample File (Instant)

A ready-to-use harmonic sample audio file is pre-included in the repository at `samples/sample.wav`.
You can test the running API immediately with:

```bash
curl -F audio=@samples/sample.wav http://localhost:8000/analyze
```

---

## Option 1: Mozilla Common Voice (recommended for eval)

Mozilla Common Voice is the correct dataset for this service because it includes
age, gender, and accent metadata — the same labels our model predicts.

1. Go to: https://commonvoice.mozilla.org/en/datasets
2. Select **English** (for baseline) and **Hindi** (for India calibration)
3. Download the validated clips (requires a free account)
4. Extract to a local directory and point `eval/run_eval.py` at it:

```bash
python eval/run_eval.py \
  --dataset-path /path/to/cv-corpus \
  --locales en hi \
  --max-samples 200
```

A single clip from the `clips/` folder makes a good smoke test:
```bash
curl -F audio=@/path/to/cv-corpus/en/clips/sample.mp3 \
     http://localhost:8000/analyze
```

---

## Option 2: LibriSpeech (quick download, no labels needed for smoke test)

LibriSpeech provides clean read speech. No age/gender labels, so it can't be used
for accuracy evaluation — but it works fine for verifying the service is running.

1. Download a small subset: https://www.openslr.org/12/
   - `test-clean.tar.gz` is the smallest (~346 MB)
2. Any `.flac` file from the archive works:

```bash
curl -F audio=@path/to/1234-5678-0001.flac \
     http://localhost:8000/analyze
```

---

## Option 3: Generate a test clip via ffmpeg

If ffmpeg is installed:
```bash
ffmpeg -f lavfi -i "sine=frequency=300:duration=5" -ar 16000 -ac 1 sample_sine.wav
curl -F audio=@sample_sine.wav http://localhost:8000/analyze
```

Note: a pure sine wave will be classified as `audio_quality: "insufficient"` because
webrtcvad doesn't detect it as voiced speech. This is correct behaviour — use a real
speech recording for a meaningful prediction.

---

## Privacy note

Do not commit real caller audio to the repository. Sample files in this directory
should be either generated synthetic audio or clips from open-licensed datasets
(Common Voice is CC0 licensed). See `DPDP_PRIVACY.md` for the full privacy posture.
