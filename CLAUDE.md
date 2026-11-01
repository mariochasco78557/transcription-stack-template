# CLAUDE.md — Transcription 24/7

Project guide for future Claude Code sessions. Read `docs/ARCHITECTURE.md` before changing
anything: it is the binding contract (module signatures, JSON schemas, state machine, config keys).
`README.md` is the human-facing quickstart.

## What this is

Self-hosted meeting-transcription pipeline (a Fireflies-style replacement). Audio → Markdown meeting notes in Google Drive.
One `systemd --user` worker, strictly one job at a time, stages:

```
received → normalized → transcribed → diarized → merged → summarized → delivered
            ffmpeg      faster-       sherpa-    merge.py  claude -p    render.py
                        whisper       onnx                 (subscription) + indices.py
```

`jobs.state` is the **last completed stage**; each stage writes `data/jobs/<id>/<stage>.json`, so
any crash resumes from the last artifact. Plus `failed` (terminal until `t247 retry`) and
`summary_pending` (delivered without a summary; the worker retries it, `t247 resummarize` forces
it).

Packages: `config/paths/ids/store/models/logging_setup/cli` (core) · `audio/asr/diarize/merge`
(pipeline) · `summarize/prompts` · `render/indices` · `api/web/inbox/notify` (ingestion) ·
`worker/doctor` + `ops/` + `tests/` (integration).

## Hard boundaries — do not cross

- **Never write outside the owned tree.** Every write goes through `paths.guard_write()`, which
  allows only `archivo.raiz` (`…/Transcripciones_24_7/`), `inbox.ruta` and `data/`, and hard-denies
  any resolved path containing the segment `_Protected_Archive` (a pre-existing archive,
  owned by another system) or `_Protected_Private`. If you add code that touches the filesystem, route
  it through `guard_write` — no exceptions, and `tests/test_paths_guard.py` must stay green.
- **Never read or print secrets.** `~/.config/transcription247/{api_token,telegram.json}` are
  `0600`; `config.py` refuses any wider mode. `ops/install.sh` creates the token without printing
  it — keep it that way. No credentials in the repo, ever.
- **Tailnet only.** `api.host` is the Tailscale IP. `0.0.0.0`/`::` is refused at config load *and*
  at `serve()`. There is no override flag and none may be added.
- **The summarizer must stay on the subscription.** `summarize.call_claude` scrubs
  `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` from the child env and must never use `--bare` (that
  would bill per call, §16 D10).
- **Don't commit** unless the owner asks. `data/`, `models/`, `bench/audio/` and `.venv/` are
  gitignored and must stay that way.

## Resource rules (shared 16 GB VPS, past OOM incident)

VS Code and several Claude Code sessions share this box. Slow and steady.

- Check `free -h` first; if available RAM < ~3 GB, wait. Also check
  `systemctl --user list-units --type=service --state=running 'run-*'` — another session may be
  benchmarking.
- **Anything heavy runs outside the editor's process tree**, in a capped transient unit:
  ```bash
  systemd-run --user --collect --wait --pipe -p Nice=19 -p CPUSchedulingPolicy=idle \
    -p IOSchedulingClass=idle -p CPUQuota=400% -p MemoryMax=3G -p MemorySwapMax=0 \
    -p OOMScoreAdjust=500 --working-directory="$PWD" <cmd>
  ```
  Measured for a 2-minute clip: ~57 s wall, peak RSS ~1.65 GB (ASR dominates; diarization ≤ 440 MB).
  A 1 h recording is ≈ 20 min. Never run two pipeline jobs at once.
- Real `claude` calls cost real subscription quota: mock `summarize.summarize` in tests. Watch out
  for `monkeypatch.undo()` — it restores the *real* function; inject failures through the fake
  instead (see `tests/test_worker_integration.py::Calls.break_summary`).
- Don't load whole transcripts, journals or model files into context; grep and read ranges.

## Commands

```bash
uv sync                      # deps
uv run pytest -q             # 159 tests, no models, no network, ~3 s
uv run t247 doctor           # mount, models, claude CLI, tailnet, RAM, secrets, units
uv run t247 status           # queue
uv run t247 enqueue FILE --title "…" --speakers 2
uv run t247 retry JOB_ID | resummarize JOB_ID | reindex
ops/install.sh               # idempotent: secrets, Drive folders, units, enable --now, doctor
ops/uninstall.sh             # units only — never deletes data or Drive content
journalctl --user -u t247-worker -f
systemctl --user status t247-worker t247-api
```

## Things that will bite you

- **systemd escaping.** `ExecStart=` needs `\x20` for each space in the project path;
  `WorkingDirectory=` needs the spaces **literally** (`\x20` there is not unescaped and quoting
  yields an empty value). Verify with `systemd-analyze --user verify` *and*
  `systemctl --user show -p WorkingDirectory -p ExecStart <unit>`.
- **`tailscaled` is a system unit**, so `t247-api.service` (a user unit) cannot order itself after
  it. It fails fast, restarts on failure, and `StartLimitIntervalSec=0` keeps it retrying.
- **`Store` is shared across threads** (worker loop + inbox poller): the connection is
  `check_same_thread=False` and every method holds an `RLock`. Keep new methods decorated with
  `@_locked`, and hold the lock across a whole `BEGIN IMMEDIATE … COMMIT`.
- **`paths`' roots are module globals.** Call `paths.configure(cfg)` once at process start; in
  tests rely on the autouse `_isolate_paths_roots` fixture in `tests/conftest.py` instead of
  leaking roots between tests.
- **rclone FUSE mount.** Writes use a dot-prefixed temp + `os.replace`, no directory fsync, no
  hardlinks. If `archivo.raiz` is missing or unwritable the worker *holds* the job (never fails it)
  and retries later.
- **`t247 doctor` writes exactly one probe file** into the Drive root and removes it — that is the
  only way to tell "mount listed" from "mount writable".

## Ownership notes

`diarize.py`, the `[diarization]` config section and its dataclass, and `bench/` belong to the
diarization tuning work. `docs/CLIENTES_IPHONE_MAC.md` belongs to whoever writes the client
recipes. Tolerate new keys appearing under `[diarization]`; don't edit those files.
