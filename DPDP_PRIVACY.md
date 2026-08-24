# Privacy & DPDP Act 2023 Notes

This service processes voice audio of callers (drivers, dispatchers, customers) who
have not separately consented to demographic inference — they consented, at most, to
the underlying call itself. Under India's Digital Personal Data Protection Act, 2023,
voice recordings and attributes derived from them are personal data, and processing
requires a lawful basis (consent or a permitted "legitimate use") and purpose
limitation — the data can't quietly be used beyond the stated purpose.

This document is a submission artifact, not legal advice, and doesn't replace review
by Dialflo's own counsel — but it should read like someone who understands the
obligation exists, not just that "we don't store audio" is a compliance strategy on
its own.

## What this service actually does

- Audio is held in memory only for the duration of a single request/stream.
- No audio bytes, no derived embeddings, and no raw waveform are written to disk, to
  logs, or to any persistent store at any point in the pipeline — including on error
  paths (a common leak point: naive exception handlers that dump the request payload
  into an error log). `logging_config.py` is deliberately structured so the only
  fields it's physically possible to log are scalars (ids, timings, labels,
  confidences) — there's no code path where an audio buffer reaches the logger.
- Predictions are returned to the caller of this API and not persisted by this
  service — what the calling system (Dialflo's own call platform) does with the
  result afterward is outside this service's boundary, and the README should say so
  explicitly rather than implying end-to-end compliance this service doesn't control.

## What still needs a decision at the product level (flag, don't solve)

- **Lawful basis**: is caller consent to demographic inference covered by existing
  call-recording disclosures, or does it need its own notice? This is a product/legal
  decision, not something this service can resolve — the README should say this
  explicitly rather than assume it away.
- **Purpose limitation**: this service should be positioned, and used, only for
  real-time conversational personalization (tone/greeting) — not fed into any
  downstream profile, CRM record, or analytics store without a separate lawful basis
  for that secondary use.
- **Data principal rights**: since nothing is stored here, there's nothing to
  correct/delete/export from this service specifically — but that only holds as long
  as no downstream system persists the output against a contact record. Worth a
  one-line caution in the README so a reviewer doesn't assume the whole system is
  compliant just because this microservice is stateless.

## Why this section matters for the submission

Most take-home submissions for a service like this will say "we don't store audio,
so we're privacy-safe" and stop there. That's necessary but not sufficient — the
honest answer is "this service is a small, compliant piece of a system whose overall
compliance depends on decisions made outside this service's boundary," and saying
that plainly is more credible than overclaiming.
