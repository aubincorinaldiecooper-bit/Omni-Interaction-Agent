# Handover — Gander media lifecycle fix

Branch: `fix/gander-media-lifecycle`, cut from `main` at `cf43838`.
Pull request: https://github.com/aubincorinaldiecooper-bit/Omni-Interaction-Agent/pull/1

Work to the accepted lifecycle audit (`gander-lifecycle-audit.md`). Every
accepted item is implemented. What is not done is listed under "Still open".

## What a person will notice

**Your screen share stops when you stop it, and not before.** Previously
almost anything going wrong between the browser and the server ended the
operating system's capture: a dropped WebSocket, a slow acknowledgement, a
control message the runtime did not understand. The sharing bar vanished and
getting it back meant granting permission again. Capture and connection are
now separate. The share stays up, the connection reconnects underneath it,
and the status reads "reconnecting" rather than going dark.

**"Model is busy" when nothing is running.** Starting a session used to wait
for the Brain to warm before Gander would see or hear, and shutting one down
held the single model slot through slow network teardown. Gander now becomes
ready first and warms the Brain behind that; the slot is freed the moment the
model session closes. Stop then Start works immediately.

**A hiccup does not lose your conversation.** A duplex socket that drops is
held for a grace window with its model session intact, and the browser
re-binds to the same conversation. The screen channel stays open throughout.

**One bad frame no longer ends the share.** It is reported and skipped.

## Accepted audit items

| # | Item | Where |
|---|---|---|
| 1 | Split capture lifecycle from transport | `static/video.js` |
| 2 | Reconnect transport without reacquiring the stream | `static/video.js` |
| 3 | Only `fatal:true` errors end the session | `static/app.js`, `online_duplex.py` |
| 4 | Server reconnect grace, same-session resume | `online_duplex.py` |
| 5 | Bad frames emit a dropped-frame event and continue | `online_duplex.py` |
| 6 | `ready` before Brain warmup, `brain.status` published | `online_duplex.py` |
| 7 | Detect client disconnect during startup | `online_duplex.py` |
| 8 | Release the model lock before slower cleanup | `online_duplex.py`, `task_tools_online.py` |
| — | Duplex reconnect with the same `session_id` (audit 8, client) | `static/app.js` |
| — | `checkRuntime` retry with backoff (audit 9) | `static/app.js` |
| — | `waitForRuntimeRelease` polling and busy retry (audit 10) | `static/app.js` |

## Tests

Server — 16 cases, Thinker and worker gateway stubbed, no torch or GPU:

```
pip install -e ./gander_runtime[test]
pytest gander_runtime/tests
```

Client — 8 cases driving `video.js` in a stubbed browser:

```
node --test minicpm_ft/tests/client/video-lifecycle.test.mjs
```

Both suites were run repeatedly rather than once: the server suite 15
consecutive times with no failures, the client suite checked against the bug
it describes (reverting the transport-loss path fails exactly the two capture
assertions).

## Still open

- **No live run.** Nothing here has been exercised against a real model, a
  real browser or Modal. Every test uses stubs. The audit's manual checklist
  (section 6, "Manual live check") has not been performed.
- **Frame files accumulate.** Every context-sampled frame is written under
  `media_dir` and nothing deletes it — no session-end cleanup, no retention
  policy. Left out deliberately; it does not block session cleanup. Needs a
  product decision on what a session may refer back to.
- **Grace window length is a guess.** `reconnect_grace_sec` defaults to 15s,
  the audit's proposal. It has not been checked against Modal's websocket
  idle behaviour, which the audit also lists as open.
- **Which Thinker exceptions are fatal.** Currently any unhandled exception in
  the duplex handler is `fatal: true` and everything else is recoverable. The
  audit wants this narrowed once there is a real traceback to look at.
- **Whether the Clipit Modal wrapper carries local patches** to these files is
  still unverified, as the audit noted.
- **A fast reconnect can be told "busy".** If the browser reconnects before
  the server finishes parking the session, it gets the busy reply instead of
  resuming. The client retries on busy so it recovers, but the first attempt
  is wasted. The tests wait for `resumable_sessions` in `/health` to avoid it.
- **A Start straight after a disconnect during the model open can be told
  "busy".** The slot is held until the abandoned open finishes and its session
  is closed, which is correct — releasing earlier would let a second session
  build against the same model. The client retries, so it recovers.

## Things found while implementing, beyond the audit

- `stop_speech_output_pump` caught only `(WebSocketDisconnect, RuntimeError)`
  while `_WebSocketOutbox._raise_if_failed` re-raises whatever the writer task
  stored, which its `except Exception` branch preserves as any type. An
  escaping exception there would have skipped the lock release. Audit item 8's
  reordering plus a nested `finally` closes this whatever the type. Not
  demonstrated firing in practice.
- The talker pump only checked its stop flag when a poll came back empty, so
  it could run for as long as the model kept talking while teardown waited on
  it. It now drains and finishes, and the wait is bounded.
- A handler that ends in cancellation rather than disconnection cannot await
  anything in its teardown — the first `await` re-raises. Parking is therefore
  recorded synchronously and the grace timer does the waiting. This was found
  by a test failing about one run in six.
- The grace timer is held on the runtime. A task with no strong reference can
  be garbage-collected while pending, skipping its teardown entirely.

## Next step

Review and merge the pull request. Per the agreed order, the Clipit versioned
Modal deploy does not start until this is merged and the fork is pinned to an
exact SHA. The SearXNG discovery refactor, the browser runtime and the
four-scout work are CLIPIT-side and come after that; Gander stays the
audiovisual runtime only.

Before merging, the manual live check is worth running, since nothing here has
been seen working against a real browser.
