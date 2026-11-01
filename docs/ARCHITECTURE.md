# Transcription 24/7 — Architecture

**v1.0 (design freeze) · 2026-09-19 · Owner: project maintainer · `version_modulo = "1.0.0"`**

Zero-monthly-cost, self-hosted replacement for Fireflies.ai, single user, on a shared Debian 13 VPS
(8 vCPU / 16 GB shared with VS Code and several Claude Code sessions). Audio in → Markdown meeting
notes in Google Drive, in exactly the format an existing Fireflies archive uses.

This document is the contract: engineers implement against it without talking to each other.
Unspecified details are chosen here and recorded in §16 *Decisions*.

---

## 1. Overview and data flow

Two ingestion paths feed one SQLite queue. One worker, one job at a time, strictly serial. Each
stage loads its model, writes a JSON artifact, and releases the model before the next loads — peak
RAM is `max(stage)`, never the sum.

```
 iPhone (Voice Memos)                Mac (Drive desktop / launchd)
   ├─ Shortcut ─ POST /upload ──┐      ├─ curl ─ POST /upload ──┐
   └─ Save to Drive ─┐          │      └─ Drive sync ─┐         │
                     ▼          ▼                     ▼         ▼
          _INBOX_AUDIO/ (rclone FUSE)            t247-api.service
                     │   poller thread (15 s, stability check)   (uvicorn, tailnet only)
                     └───────────────┬──────────────────────────┘
                                     ▼
                       SQLite queue — data/t247.db
                                     ▼
                  t247-worker.service — ONE job at a time
  received ─► normalized ─► transcribed ─► diarized ─► merged ─► summarized ─► delivered
     copy      ffmpeg 16k    faster-whisper  sherpa-onnx  merge    claude -p     renderer
   + sha256    mono wav      large-v3-turbo  diarization  algo   (subscription)  + indices
     └─── data/jobs/<id>/{normalized,asr,diarization,turns,summary}.json ───┘
                                     ▼
       /home/USER/GoogleDrive/Transcripciones_24_7/
         <Cliente>/AAAA-MM/AAAA-MM-DD HHhMM - Título.md
         <Cliente>/INDICE.md · INDICE.md · indice_transcripciones.json · LEEME.md
                                     ▼                     └─► Telegram notification
```

Summary failure never blocks delivery: the note ships with a `Resumen pendiente` marker, the job
lands in `summary_pending`, and `t247 resummarize` fixes it later.

---

## 2. Repository layout

`[P1]`..`[P5]`/`[INT]` = owning work package (§14). `(exists)` = owned by another agent, **do not touch**.

```
Transcription_24_7/
├── pyproject.toml  (exists) deps + console script     ├── uv.lock (exists)
├── README.md       (exists, empty) [INT] quickstart   ├── .gitignore (exists)
├── bench/          (exists) benchmarks; RESULTS.md feeds the §5 defaults marked "# bench"
├── models/         (exists, gitignored) ASR + diarization model files
├── config.toml                        [P1] all non-secret configuration (§5)
├── data/           [runtime, gitignored] t247.db · jobs/<id>/ · tmp/ · logs/t247.log
├── docs/ARCHITECTURE.md  this file    ├── docs/CLIENTES_IPHONE_MAC.md [INT] client recipes (§13.1)
├── ops/t247-worker.service            [INT] systemd --user unit (§11)
├── ops/t247-api.service               [INT] systemd --user unit (§11)
├── ops/install.sh · ops/uninstall.sh  [INT] install/enable/remove units (§11)
├── src/transcription247/
│   ├── __init__.py    (exists) [P1] rewrite: __version__ + main() -> cli.main
│   ├── cli.py         [P1] argparse entrypoint, dispatches to the other packages
│   ├── config.py      [P1] load/validate config.toml + secrets dir
│   ├── paths.py       [P1] path guard, sanitization, Drive layout, time/date formatting
│   ├── ids.py         [P1] ULID job ids, sha256 helpers
│   ├── logging_setup.py [P1] journald + RotatingFileHandler
│   ├── store.py       [P1] SQLite schema, job CRUD, state machine, leases
│   ├── models.py      [P1] dataclasses mirroring every stage artifact
│   ├── audio.py       [P2] ffmpeg normalize + ffprobe metadata
│   ├── windows.py     [P2] long-audio window planning, shared by asr.py and diarize.py (D18)
│   ├── asr.py         [P2] faster-whisper wrapper
│   ├── diarize.py     [P2] Diarizer protocol + sherpa-onnx backend
│   ├── merge.py       [P2] word->speaker merge, smoothing, turn grouping (§7)
│   ├── summarize.py   [P3] headless Claude CLI client, map-reduce, validation (§8)
│   ├── prompts.py     [P3] prompt templates + JSON schema text
│   ├── render.py      [P4] Markdown note renderer + atomic write (§9)
│   ├── indices.py     [P4] INDICE.md, client indices, JSON catalog, LEEME.md (§10)
│   ├── api.py         [P5] FastAPI app: /upload, /jobs, /jobs/{id}, /healthz, /
│   ├── web.py         [P5] single-page HTML (Spanish, no JS framework)
│   ├── inbox.py       [P5] Drive inbox poller thread
│   ├── notify.py      [P5] Notifier protocol + Telegram backend
│   ├── worker.py      [INT] daemon loop, stage orchestration, pre-flight, resume, delivery
│   └── doctor.py      [INT] environment checks
└── tests/{test_merge,test_render_golden,test_paths,test_store_resume}.py + fixtures/  [INT]
```

---

## 3. Module specifications

All artifact timestamps are **float seconds from start of audio**. Artifacts are UTF-8 JSON,
`ensure_ascii=False`, written atomically (temp + `os.replace`), each carrying `"schema": <int>` —
bump on incompatible change; readers reject unknown values.

### 3.1 `paths.py` [P1]

```python
OUTPUT_ROOT: Path   # config archivo.raiz
INBOX_ROOT: Path    # config inbox.ruta
DATA_ROOT: Path     # <project>/data

class PathGuardError(RuntimeError): ...

def ensure_inside(root: Path, candidate: Path) -> Path
    """Resolve(strict=False) and assert candidate is root or below it, else PathGuardError."""

def guard_write(candidate: Path) -> Path
    """ensure_inside against OUTPUT_ROOT / INBOX_ROOT / DATA_ROOT; additionally hard-deny any
    resolved path containing the segment '_Protected_Archive' (a protected sibling archive) or
    '_Protected_Private'. EVERY filesystem write in the project goes through this function."""

def sanitize_title(raw: str, max_len: int = 120) -> str
    """NFC-normalize; drop control chars; map / \\ : * ? " < > | to '-'; collapse whitespace;
    strip leading dots and trailing dots/spaces; truncate on a word boundary;
    '' -> 'Reunión sin título'."""

def note_path(cliente_folder: str, start: datetime, titulo: str) -> Path
    """OUTPUT_ROOT/<cliente_folder>/<YYYY-MM>/<YYYY-MM-DD HHhMM> - <titulo>.md, guarded."""

def extract_speaker_hint(name: str) -> tuple[str, int | None]
    """Find a case-insensitive [Np] or (Np) token (N in 1-10) anywhere in `name`; return
    (name with the matched token removed, N) or (name unchanged, None) if absent/out of range.
    Used by api.py and inbox.py (§6) before sanitize_title, so the token never reaches
    source_name; the caller sets jobs.speakers_user from N only when no explicit value was
    already given."""

def fmt_ts(seconds: float) -> str      # 'MM:SS' below 3600 s, else 'H:MM:SS'; zero-padded
def fmt_long_date_es(d: date) -> str   # 'martes 1 de septiembre de 2026'; module-level Spanish
                                       # DIAS/MESES tables — never the C locale
```

### 3.2 `ids.py` [P1]

```python
def new_job_id() -> str          # 26-char lowercase Crockford ULID, time-sortable
def sha256_file(p: Path) -> str  # streaming, 1 MiB chunks, lowercase hex
def sha256_text(s: str) -> str
```

### 3.3 `audio.py` [P2] → `normalized.json`

```python
SUPPORTED_EXT = {".m4a",".mp3",".wav",".aac",".ogg",".opus",".flac",".mp4",".mov",".webm",".caf"}
class AudioError(RuntimeError): ...

def probe(src: Path) -> dict            # ffprobe -v quiet -print_format json -show_format -show_streams
def creation_time(src: Path) -> datetime | None   # format.tags.creation_time -> Europe/Madrid
def normalize(src: Path, dst_wav: Path) -> NormalizedAudio
    """ffmpeg -nostdin -hide_banner -loglevel error -i <src> -vn -ac 1 -ar 16000
       -acodec pcm_s16le -f wav <dst_wav>.
       AudioError on non-zero exit or duration < config.audio.min_duracion_s."""
```

```json
{"schema":1,"job_id":"01k5r2m8q7v3x9b0c1d2e3f4g5","source_name":"Nueva grabación 12.m4a",
 "source_sha256":"87aa1a29…","source_bytes":18234123,"wav_path":"data/jobs/01k5…/audio.wav",
 "duration_s":4712.48,"sample_rate":16000,"channels":1,"container":"mov,mp4,m4a","codec":"aac",
 "creation_time":"2026-09-19T10:15:00+02:00","start_source":"metadata"}
```

`start_source` ∈ `"upload" | "metadata" | "filename" | "mtime"` (resolution order, §9.2).

### 3.4 `asr.py` [P2] → `asr.json`

```python
class AsrError(RuntimeError): ...

class Transcriber:
    def __init__(self, cfg: AsrConfig, *, pipeline: PipelineConfig | None = None) -> None: ...
    def __enter__(self) -> "Transcriber": ...         # loads WhisperModel
    def __exit__(self, *exc) -> None: ...             # del model; gc.collect()
    def plan(self, wav: Path) -> list[Window]: ...    # windows.plan_from_config()
    def transcribe(self, wav: Path, *, partial_path: Path | None = None) -> AsrResult: ...

def run(wav: Path, cfg: AsrConfig, *, pipeline: PipelineConfig | None = None) -> AsrResult
```

`WhisperModel(cfg.model_path, device="cpu", compute_type="int8", cpu_threads=cfg.cpu_threads,
num_workers=1, local_files_only=True)`, then `model.transcribe(str(wav), beam_size=cfg.beam_size,
vad_filter=True, vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms,
"speech_pad_ms": cfg.vad_speech_pad_ms}, word_timestamps=True, condition_on_previous_text=False,
language=cfg.language or None)`. Language: use `cfg.language` when set; else auto-detect and, if the
detected language is not in `idiomas_permitidos` (`["es","en"]`), **re-run with `language="es"`** and
set `language_forced: true`. `__exit__` must free the model before the diarizer loads.

```json
{"schema":1,"job_id":"01k5…","language":"es","language_probability":0.9912,
 "language_forced":false,"duration_s":4712.48,"model":"faster-whisper-large-v3-turbo-int8",
 "segments":[{"id":0,"start":1.02,"end":6.84,"text":" Hola. En septiembre…",
              "avg_logprob":-0.21,"no_speech_prob":0.004}],
 "words":[{"start":1.02,"end":1.31,"word":" Hola.","probability":0.98},
          {"start":1.44,"end":1.62,"word":" En","probability":0.99}]}
```

`words` is one flat list over the whole recording, ordered by `start`, keeping faster-whisper's
leading space in `word`. Empty `words` → **0 speech detected** (§13, #7).

#### Long audio — windowing (D18)

`model.transcribe()` runs Silero VAD over the **entire** array before decoding a single segment, so
its footprint grows with the recording (measured ~10 MB per audio-minute of VAD state on top of
~3.8 MB/min of decoded float32). Passing a 60-min wav was OOM-killed 17 s in; only ≤ 30 min fitted.

The stage therefore loads the model **once** and transcribes **window by window**:

1. `windows.plan_from_config(wav, cfg.pipeline)` → contiguous, non-overlapping `Window`s of at most
   `pipeline.window_max_s` (900 s), each cut at the longest low-energy pause found in the last
   `pipeline.window_cut_search_s` (90 s) of the window, or hard at `window_max_s` when the tail is
   continuous speech. The scan is a block-wise RMS over ~90 s read in chunks — the file is never
   loaded whole. A recording shorter than `window_max_s` yields **one** window and takes the
   original code path (the wav path handed straight to faster-whisper), unchanged.
2. Each window's samples are read on demand (`soundfile` `start=`/`frames=`), so at most one window
   is resident; the buffers are dropped and `release_arenas()` called before the next.
3. Every segment and word timestamp is shifted by the window's `start_s`, and segment `id`s keep
   counting across windows, so `asr.json` is *exactly* the same shape it always was — one flat
   global timeline. The only addition is `"windows": [[start, end], …]`, the plan actually used
   (absent in artifacts written before windowing; consumers ignore it).
4. **Language** is detected on the first window that actually produced words (still restricted to
   `idiomas_permitidos`, still falling back to `idioma_fallback` with `language_forced: true`) and
   then **pinned** for every remaining window. Re-detecting per window can flip on a window of
   silence or noise and split one recording across two languages.
5. The progress callback fires at least once per window; the worker uses it to **renew the job
   lease** (§4), which a 2 h recording needs — it spends longer in this one stage than the old
   between-stages renewal assumed.

With `pipeline.asr_partial_resume` (default `true`) the stage rewrites `asr.partial.json` after
every window, so a worker killed 90 min into a 2 h recording restarts at the next window instead of
at zero. It is only trusted when its persisted window plan matches the one just recomputed, so a
config change starts over rather than splicing two timelines; the worker deletes it once `asr.json`
is written. `pipeline.window_max_s = 0` disables windowing entirely (the pre-windowing behaviour).

### 3.5 `diarize.py` [P2] → `diarization.json`

```python
class Diarizer(Protocol):
    def __enter__(self) -> "Diarizer": ...
    def __exit__(self, *exc) -> None: ...
    def diarize(self, wav: Path, duration_s: float) -> list[SpeakerSegment]: ...

class SherpaOnnxDiarizer:
    def __init__(self, cfg: DiarizationConfig) -> None: ...

def get_diarizer(cfg: DiarizationConfig) -> Diarizer
    """Factory on cfg.backend ('sherpa_onnx' today, 'pyannote' reserved); else ValueError."""
```

Backend: `sherpa_onnx.OfflineSpeakerDiarization` with `OfflineSpeakerDiarizationConfig(
segmentation=…pyannote(model=cfg.segmentation_model), embedding=…(model=cfg.embedding_model,
num_threads=cfg.num_threads), clustering=FastClusteringConfig(num_clusters=cfg.num_clusters,
threshold=…), min_duration_on=…, min_duration_off=…)`; `num_clusters = -1` means the speaker count
is unknown. Audio read with `soundfile` as float32 mono 16 k.

`cfg.num_clusters` (`DiarizationConfig.num_clusters`) is per-job, not a single global constant:
`diarize.py` itself only ever reads `cfg.num_clusters` — it has no knowledge of jobs. The worker
(`[INT]`) builds this per-job `DiarizationConfig` as `dataclasses.replace(config.diarization,
num_clusters=job.speakers_user)` when the job carries a `speakers_user` value (§4, §6), else it
passes `config.diarization` unchanged (`num_clusters=-1`). A known speaker count is always cheaper
and reliably correct (bench: exact on every clip); everything below is for when the caller doesn't
know — which, Fireflies-style, is the normal case.

**Unknown speaker count — `auto_count = "overcluster"` (default).** A single cosine threshold on
sherpa's own per-window embeddings is not usable: the same setting that gives 2 speakers on one
clip gives 8-18 on the next (bench/RESULTS.md, §16 D15). `diarize.py` instead *over-clusters and
merges*:

1. sherpa runs with `threshold = cfg.overcluster_threshold` (0.15) — deliberately far too low, so
   the output is a heavily over-split but locally pure partition (3-17 "over-clusters" on the dev
   set) rather than an answer.
2. Every returned segment of at least 0.5 s is re-embedded with `cfg.count_embedding_model`
   (3D-Speaker eres2net, trained on Mandarin, not VoxCeleb-English) and the segments of each
   over-cluster are pooled into one duration-weighted, L2-normalised centroid. The count decision
   is therefore taken on tens of seconds of speech per centroid, not on ~1 s windows.
3. The centroids are merged bottom-up (cosine distance, weighted average linkage, numpy only) and
   the dendrogram is cut at `cfg.merge_threshold` (0.475).
4. Clusters holding less than `cfg.min_cluster_share` (3 %) of the speech are folded into their
   nearest surviving cluster, and the result is capped at `cfg.max_speakers` (8). Over-clusters too
   short to seed the decision are then assigned to the nearest final centroid.
5. Segments are relabelled and consecutive same-speaker segments closer than `min_duration_off` are
   glued back together.

`auto_count = "threshold"` restores the old single-`cluster_threshold` behaviour and exists only as
an escape hatch. The known-count path (`num_clusters > 0`) is untouched by all of this.

#### Long audio — windowing and global speaker linking (D18)

`OfflineSpeakerDiarization.process()` holds the whole waveform plus its per-frame segmentation
output: ~54 MB per audio-minute, i.e. ~3.4 GB at 60 min and ~6.6 GB at 120 min. The stage runs on
the **same `Window` plan as the ASR** (`windows.plan_from_config`, so the two always agree) in two
passes, never holding both ONNX sessions or more than one window of audio:

1. **Segmentation pass.** sherpa runs per window with `num_clusters = -1` and
   `overcluster_threshold`, producing *window-local* cluster ids. Timestamps are shifted onto the
   global timeline immediately; the samples are freed before the next window. The segmentation
   session is then released.
2. **Embedding pass.** `cfg.count_embedding_model` re-embeds each window's segments and pools them
   into **one duration-weighted centroid per `(window, local id)`**. Only these centroids survive a
   window — a few hundred 192-float rows whatever the duration. Audio and per-segment embeddings do
   not.
3. **One global merge.** `link_windows()` runs the *same* recipe as the single-window path over all
   windows' centroids at once (cosine distance, weighted average linkage, cut at `merge_threshold`,
   tiny-cluster absorption on the **global** speech share, `max_speakers` cap) and returns
   `(window, local id) → global speaker id`. Adjacent same-speaker segments are then glued across
   window joins with `min_duration_off`.

Why one global merge rather than per-window clustering stitched together afterwards: stitching has
no way to recognise a speaker who is silent across a join, and its errors compound window after
window. Merging the centroids in a single pass means the decision always sees the whole recording,
at constant memory. Judging `min_cluster_share` globally matters for the same reason — a speaker
with 4 s in a 2 h meeting is noise, even if those 4 s are 20 % of one window.

**Known count with more than one window** takes the same route but cuts the global dendrogram at
exactly `k` (`_cut_k`), skipping the tiny-cluster and `max_speakers` rules: the caller asked for a
count and gets it. Forcing `k` *inside* each window would be wrong — a speaker can be silent for a
whole window, and the window would then split someone else in two to fill the quota.

A recording that fits in one window skips all of the above and takes the original path verbatim.
This is not an approximation: pooling per window and handing `estimate_speakers()` one row per
cluster is arithmetically identical to letting it pool the per-segment rows itself (a positive
scalar vanishes under L2 normalisation), which `tests/test_diarize_linking.py` pins directly and
`bench/diar_e2e.py` confirms end to end (7/8 unchanged).

```json
{"schema":1,"job_id":"01k5…","backend":"sherpa_onnx","num_speakers_detected":3,
 "segments":[{"start":0.98,"end":42.31,"speaker":0},{"start":42.31,"end":55.02,"speaker":1}],
 "windows":[[0.0,884.32],[884.32,1771.06]]}
```

`speaker` is an opaque cluster id, **not** a display number. Segments sorted by `start`, non-overlapping.

### 3.6 `merge.py` [P2] → `turns.json` (algorithm in §7)

```python
def assign_speakers(words: list[Word], segments: list[SpeakerSegment], cfg: MergeConfig) -> list[int]
def smooth(word_speakers: list[int], words: list[Word], cfg: MergeConfig) -> list[int]
def group_turns(words: list[Word], word_speakers: list[int], cfg: MergeConfig) -> list[Turn]
def renumber(turns: list[Turn]) -> tuple[list[Turn], list[str]]   # -> (turns, participantes)
def merge(asr: AsrResult, diar: list[SpeakerSegment], cfg: MergeConfig) -> TurnsResult
```

```json
{"schema":1,"job_id":"01k5…","language":"es","duration_s":4712.48,
 "participantes":["Speaker 1","Speaker 2","Speaker 3"],
 "turns":[{"start":1.02,"end":121.44,"speaker":0,"speaker_label":"Speaker 1",
           "text":"Hola. En septiembre llevaremos a cabo el reparto de libros…"}]}
```

### 3.7 `summarize.py` [P3] → `summary.json` (contract in §8)

```python
class SummaryError(RuntimeError): ...

def chunk_turns(turns: list[Turn], chunk_seconds: int) -> list[list[Turn]]
def render_transcript_for_prompt(turns: list[Turn]) -> str    # "[MM:SS] Speaker 1: texto"
def call_claude(prompt: str, cfg: SummaryConfig) -> dict
    """Run the CLI, parse the envelope, strip ``` fences, json.loads the payload.
    SummaryError on timeout, non-zero exit, is_error, or unparseable JSON."""
def summarize(turns_result: TurnsResult, hints: SummaryHints, cfg: SummaryConfig) -> SummaryResult
    """Map-reduce when needed, validate, ONE retry, raise SummaryError if both fail."""
```

`SummaryHints`: `titulo_usuario: str | None`, `cliente_usuario: str | None`,
`clientes_catalogo: list[dict]`, `categorias_catalogo: list[dict]` — so Claude can only choose keys
that exist. `summary.json` = the validated object plus `schema`, `job_id`, `pending`, `chunks`, `model`.

### 3.8 `notify.py` [P5]

```python
class Notifier(Protocol):
    def send(self, text: str) -> bool: ...

class NullNotifier: ...          # when no secrets file is present
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: float = 10.0) -> None: ...
    def send(self, text: str) -> bool
        """POST https://api.telegram.org/bot<token>/sendMessage with
        {chat_id, text, parse_mode:'HTML', disable_web_page_preview:true} via urllib.request
        (stdlib only). Never raises; returns False and logs."""

def get_notifier(cfg: Config) -> Notifier
```

Success message: title, duration, client, `resumen_corto`, Drive path. Failure message: job id,
source name, stage, error. Both are best-effort and never affect job state.

---

## 4. SQLite schema, state machine, retry/resume

`data/t247.db`. On every connection: `PRAGMA journal_mode=WAL; synchronous=NORMAL;
busy_timeout=10000; foreign_keys=ON;`

```sql
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,                 -- ULID
  state TEXT NOT NULL,                 -- see state machine below
  created_at TEXT NOT NULL,            -- ISO8601 with offset
  updated_at TEXT NOT NULL,
  source_name TEXT NOT NULL,           -- original filename, sanitized
  source_path TEXT,                    -- data/jobs/<id>/source.<ext>
  origin TEXT NOT NULL,                -- 'upload' | 'inbox'
  origin_ref TEXT,                     -- inbox: absolute path of the Drive file
  audio_sha256 TEXT NOT NULL,
  duration_s REAL,
  title_user TEXT, client_user TEXT, language_user TEXT, recorded_at_user TEXT,
  speakers_user INTEGER,               -- optional known speaker count (1-10), -> diarizer num_clusters
  recorded_at TEXT,                    -- resolved start, ISO with offset
  start_source TEXT,                   -- 'upload'|'metadata'|'filename'|'mtime'
  language TEXT,
  client_key TEXT,
  client_source TEXT,                  -- 'upload'|'marker'|'claude'|'fallback'
  note_path TEXT,                      -- absolute path of the delivered .md
  attempts INTEGER NOT NULL DEFAULT 0,
  summary_attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT,
  lease_until TEXT,                    -- set while a worker owns the job
  error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(audio_sha256);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, created_at);

CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  ts TEXT NOT NULL, level TEXT NOT NULL,      -- 'info'|'warn'|'error'
  stage TEXT NOT NULL, message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
```

**States.** `received → normalized → transcribed → diarized → merged → summarized → delivered`,
plus `failed` and `summary_pending`. `state` is the **last completed stage**; a worker runs every
stage after it.

* `summary_pending` means *delivered without a summary*: the `.md` exists and the indices include it
  with `resumen_pendiente: true`. `t247 resummarize <id>` re-runs the summary, re-renders, → `delivered`.
* `failed` is terminal until `t247 retry <id>`, which zeroes `attempts`/`error` and rewinds `state`
  to the last stage whose artifact is present and parseable.

**Claim/lease.** `UPDATE jobs SET lease_until = now + worker.lease_minutes WHERE id = ? AND
(lease_until IS NULL OR lease_until < now)` in a transaction; 0 rows → another worker has it.
The lease is refreshed after each stage.

**Startup requeue.** Any job with a past `lease_until` whose state is not
`delivered`/`failed`/`summary_pending` gets `lease_until = NULL` and is re-queued.

**Stage skip / resume.** Before stage *S*, if `data/jobs/<id>/<S>.json` exists and parses with a
known `schema`, the stage is skipped and the artifact reused; a corrupt artifact is deleted and the
stage re-runs. A crash therefore resumes at the last finished stage.

**Retries.** On stage failure `attempts += 1`; below `worker.max_attempts` (3) set
`next_attempt_at = now + worker.retry_backoff_s[attempts-1]` (`[60, 300, 1800]`) and leave the state
at the last good stage; at the cap → `failed` + failure notification. Summary retries are counted in
`summary_attempts` and never push a job to `failed`.

**Selection.** `state NOT IN ('delivered','failed','summary_pending') AND (next_attempt_at IS NULL
OR next_attempt_at <= now) ORDER BY created_at ASC LIMIT 1`.

---

## 5. `config.toml` (complete, with defaults)

Committed, in the project root, **no secrets**. Values marked `# bench` are provisional and must be
replaced from `bench/RESULTS.md`.

```toml
[general]
zona_horaria = "Europe/Madrid"
version_modulo = "1.0.0"
idiomas_permitidos = ["es", "en"]
idioma_fallback = "es"

[paths]
data_dir = "data"                  # relative to project root
models_dir = "models"
secrets_dir = "~/.config/transcription247"

[audio]
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"
sample_rate = 16000
min_duracion_s = 1.0
max_bytes = 1073741824             # 1 GiB
extensiones = ["m4a","mp3","wav","aac","ogg","opus","flac","mp4","mov","webm","caf"]

[pipeline]                          # §3.4/§3.5 "Long audio", D18
window_max_s = 900                 # measured: keeps peak RSS flat in the duration. 0 = no windowing
window_cut_search_s = 90           # tail of each window scanned for a low-energy pause to cut at
window_min_s = 60                  # shortest window emitted; a shorter tail is absorbed backwards
asr_partial_resume = true          # asr.partial.json: an interrupted ASR resumes at the next window

[asr]
model_path = "models/faster-whisper-large-v3-turbo/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/<SNAPSHOT_HASH>"
                                    # bench: the HF-cache snapshot dir actually on disk (config.json,
                                    # model.bin, tokenizer.json, vocabulary.json) — WhisperModel(cfg.model_path,
                                    # local_files_only=True) loads a CT2 model directory directly, no repo-id
                                    # resolution needed
compute_type = "int8"              # bench, confirmed
cpu_threads = 4                    # bench, confirmed
beam_size = 5                      # bench, confirmed
vad_min_silence_ms = 500           # bench
vad_speech_pad_ms = 200            # bench
language = ""                      # "" = auto-detect restricted to idiomas_permitidos

[diarization]
backend = "sherpa_onnx"
segmentation_model = "models/sherpa-onnx-diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
                                    # bench: real path on disk (fp32; model.int8.onnx also present but unbenched)
embedding_model = "models/sherpa-onnx-diarization/embedding/wespeaker_en_resnet34.onnx"
                                    # bench: real filename on disk (wespeaker, VoxCeleb-trained, resnet34);
                                    # ~30% lower RTF and ~half the peak RSS of the eres2net alternative also
                                    # present under embedding/, with equal-or-better speaker-count accuracy
num_threads = 4                    # bench, confirmed
num_clusters = -1                  # -1 = auto (auto_count below); set to jobs.speakers_user by the worker
                                    # when the caller supplied it (§6, §14) — always honoured over the auto path
min_duration_on = 0.30             # bench
min_duration_off = 0.50            # bench
auto_count = "overcluster"         # §3.5, D17. "threshold" = the old single-threshold behaviour
count_embedding_model = "models/sherpa-onnx-diarization/embedding/eres2net_base_zh.onnx"
                                    # bench: 3D-Speaker eres2net. Used ONLY to re-embed the over-clusters when
                                    # the count is unknown; the VoxCeleb-English embedding above barely
                                    # separates Spanish speakers (D17), this one does
overcluster_threshold = 0.15       # bench: sherpa's own clustering, set deliberately low to over-split
merge_threshold = 0.475            # bench: centre of the 0.425-0.525 plateau where 7/8 dev clips are exact
min_cluster_share = 0.03           # a cluster under 3% of the speech is absorbed by its nearest neighbour
max_speakers = 8                   # hard cap on the auto-detected count
cluster_threshold = 0.90           # auto_count = "threshold" only; 3/8 on the dev set, see bench/RESULTS.md

[merge]
max_gap_s = 2.0                    # no-overlap word: nearest segment within this gap
min_run_s = 0.60                   # smoothing: shorter runs may be absorbed
min_run_words = 2
smooth_passes = 3
turn_gap_s = 1.20                  # silence that forces a new turn
max_turn_s = 90.0                  # split long turns at the next sentence boundary

[summary]
claude_bin = "/home/USER/.local/bin/claude"
modelo = "sonnet"
timeout_s = 300
chunk_seconds = 1500               # ~25 min
max_reintentos = 1
confianza_minima_cliente = 0.70

[archivo]
raiz = "/home/USER/GoogleDrive/Transcripciones_24_7"
carpeta_fallback = "_Sin_clasificar"
renombrar_speakers = false         # config hook only; renaming is out of scope
responsables_owner = ["Owner", "(tú)"]   # §10 "Tareas abiertas del owner": responsable markers that
                                         # make a task group in a note the owner's own

[archivo.clientes.organizacion_ejemplo]
carpeta = "Organizacion_Ejemplo"
nombre = "Organización Ejemplo"
marcadores = ["ejemplo","producto","roadmap"]

[archivo.clientes.sin_clasificar]
carpeta = "_Sin_clasificar"
nombre = "Sin clasificar"
marcadores = []

[archivo.categorias]
otras = "Otras"
claustro = "Claustro"
departamento = "Departamento"
convivencia = "Convivencia"
formacion = "Formación"
tutoria = "Tutoría"
coordinacion_docencia = "Coordinación / docencia compartida"

[inbox]
ruta = "/home/USER/GoogleDrive/Transcripciones_24_7/_INBOX_AUDIO"
habilitado = true
poll_segundos = 15
estabilidad_segundos = 30          # size unchanged across 2 polls >= this apart
after_success = "delete"           # "delete" | "archive"
carpeta_procesados = "_procesados"
carpeta_error = "_error"

[api]
habilitado = true
host = "<TAILSCALE_IP>"            # tailnet IP ONLY — never 0.0.0.0
puerto = 8247
max_upload_bytes = 1073741824
titulo_web = "Transcripciones 24/7"

[worker]
mem_disponible_minima_mb = 3072
mem_espera_segundos = 60
mem_max_esperas = 60
lease_minutes = 120
max_attempts = 3
retry_backoff_s = [60, 300, 1800]
borrar_audio_tras_entrega = true
conservar_artefactos_dias = 14
idle_poll_segundos = 5

[logging]
nivel = "INFO"
fichero = "data/logs/t247.log"
max_bytes = 5242880
backups = 3
```

**Secrets** (never in the repo), directory `700`, files `600`:
`~/.config/transcription247/api_token` (one line) and `telegram.json`
(`{"token":"…","chat_id":"…"}`).

`config.py` exposes `load_config(path: Path | None = None) -> Config` (frozen dataclasses),
`Config.secret_api_token() -> str | None` and `Config.secret_telegram() -> dict | None`; both
**refuse to read a file whose mode is not 0600** — they log a warning and return `None`.

---

## 6. HTTP API

**FastAPI + uvicorn, chosen over stdlib.** Streaming multipart parsing of 1 GiB uploads is the one
genuinely error-prone piece here and Starlette already does it correctly with a spooled temp file;
`http.server` is single-threaded and would need a hand-rolled multipart parser. Cost is ~25 MB RSS
over bare Python, well inside `MemoryMax=300M` provided the body is streamed to disk and never read
into memory.

Bound to `api.host` only (§12). Single worker: `uvicorn.run(app, host=cfg.api.host,
port=cfg.api.puerto, workers=1, log_config=None, timeout_keep_alive=30)`.
Auth: `Authorization: Bearer <token>` on `/upload`, `/jobs`, `/jobs/{id}`, compared with
`hmac.compare_digest`. `/healthz` and `/` are open (already tailnet-only).

### `POST /upload`

`multipart/form-data` with file field `file`, or a raw body (`application/octet-stream` +
`X-Filename`). Optional form fields (or `X-` headers in the raw form): `title`, `language`
(`es`|`en`), `client` (a key from `archivo.clientes`), `recorded_at` (ISO 8601), `speakers`
(integer `1`-`10`, known speaker count — stored as `jobs.speakers_user` and passed to the diarizer
as `num_clusters`, §3.5; out-of-range or non-integer → 400 `bad_request`).

```bash
curl -sS -H "Authorization: Bearer $T247_TOKEN" \
  -F "file=@/tmp/reunion.m4a" -F "title=Reunión departamento" -F "client=centro_ejemplo" \
  http://<TAILSCALE_IP>:8247/upload
```
```json
{"job_id":"01k5r2m8q7v3x9b0c1d2e3f4g5","state":"received","duplicate":false,
 "source_name":"reunion.m4a","bytes":18234123}
```

Duplicate (same `audio_sha256`) → **200** `{"duplicate":true,"job_id":"<existing>",
"state":"delivered","note_path":"Centro_Ejemplo/2026-09/…md"}`; the new upload is discarded.

| Code | When | Body |
|---|---|---|
| 200 | accepted, or duplicate | as above |
| 400 | missing file, bad `recorded_at`, unknown `client`/`language`, `speakers` outside 1-10 | `{"error":"bad_request","detail":"…"}` |
| 401 | missing/wrong bearer token | `{"error":"unauthorized"}` |
| 413 | body over `api.max_upload_bytes` | `{"error":"too_large","limit":1073741824}` |
| 415 | extension not in `audio.extensiones` | `{"error":"unsupported_media_type","ext":".txt"}` |
| 503 | queue DB unreachable | `{"error":"unavailable"}` |
| 507 | free disk below 2× upload size | `{"error":"insufficient_storage"}` |

Handling: stream to `data/tmp/upload-<uuid4>.part` in 1 MiB chunks, counting bytes and aborting past
the limit, hashing SHA-256 as it goes; then `os.replace` into `data/jobs/<id>/source<ext>`. The
client filename is reduced to `Path(name).name`; before `sanitize_title` runs (both here and in the
inbox poller, §14 P5), an optional **filename speaker hint** is extracted: a case-insensitive
`[Np]` or `(Np)` token anywhere in the stem, `N` in `1`-`10` (regex `[\[(](\d{1,2})p[\])]`). If
found, and no explicit `speakers`/`--speakers` value was already given, it sets `speakers_user = N`;
the matched token (brackets/parens included) is removed from the name before `sanitize_title`, so
it never appears in `source_name`. This is the only way inbox-origin jobs (no HTTP form) can supply
a speaker count.

### `GET /jobs?limit=50&state=delivered`

```json
{"total":37,"jobs":[{"id":"01k5…","state":"delivered","created_at":"2026-09-19T10:18:03+02:00",
 "source_name":"reunion.m4a","title":"Reunión departamento","client_key":"centro_ejemplo",
 "duration_s":2881.0,"note_path":"Centro_Ejemplo/2026-09/2026-09-19 10h15 - Reunión departamento.md",
 "error":null}]}
```

### `GET /jobs/{id}`
Same object plus `"events"` (last 50 `{ts, level, stage, message}`) and `"artifacts"`
(the artifact names present). Unknown id → **404** `{"error":"not_found"}`.

### `GET /healthz` (no auth)
```json
{"ok":true,"version":"1.0.0","db":true,"inbox_mount":true,"output_mount":true,
 "queue_pending":1,"mem_available_mb":9840}
```
`ok:false` + HTTP 503 when the DB is unreachable or either Drive path is missing.

### `GET /` — single page
One self-contained HTML document from `web.py` (no framework, no CDN): Spanish UI, a
`<input type="file" accept="audio/*,video/*">` plus title/client/language fields posting to
`/upload` via `fetch`, token read from `localStorage` (prompted once), a progress indicator, and a
table of the last 20 jobs refreshed every 10 s from `/jobs`. Mobile-first: viewport meta, single
column, ≥16 px inputs so iOS does not zoom.

---

## 7. Merge algorithm

Input `asr.words` (flat, ordered) + `diarization.segments`; output `turns.json`.

**1 — Assignment by maximum overlap.** For word *w*, segment *s*:
`ov = max(0, min(w.end, s.end) − max(w.start, s.start))`; assign `argmax_s ov`, ties to the lower
index. Both lists are sorted, so walk them with a moving cursor — O(n+m), not O(n·m).
If all overlaps are 0: take the segment whose centre is nearest the word centre **if** that distance
≤ `merge.max_gap_s`; else inherit the previous word's speaker; else the first segment's speaker;
if there are no segments at all, `0`.

**2 — Smoothing of short flips.** Compress labels into runs. Absorb run *R* into its neighbours when
**all** hold: duration (`words[last].end − words[first].start`) < `merge.min_run_s`; word count ≤
`merge.min_run_words`; it has both a predecessor and a successor run; and both carry the *same*
label *L* ≠ *R*'s. Relabel *R* to *L*. Repeat until a pass changes nothing or `merge.smooth_passes`
passes have run. This removes the one-word interjections that overlap assignment produces at turn
boundaries.

**3 — Turn grouping.** Start a new turn when the label changes; **or** the silence
`words[i].start − words[i−1].end` > `merge.turn_gap_s`; **or** the current turn already spans
`merge.max_turn_s` *and* the previous word ends a sentence (stripped text ends in `.`, `?`, `!`,
`…`, `:`). The third rule keeps monologues readable without ever splitting mid-sentence.

**4 — Text assembly.** `" ".join(w.word.strip() for w in turn_words)`, then collapse repeated spaces
and delete any space before `,.;:!?…)`. Turns that end up empty are dropped.

**5 — Renumbering by first appearance.** Walk the final turns in time order: the first unseen raw id
becomes `Speaker 1`, the next `Speaker 2`, … `participantes` is that list in order. Raw ids stay in
`turn.speaker` for debugging; only `speaker_label` is rendered.

**Degenerate inputs.** No words → job fails with `no_speech` (§13, #7). Words but no segments → one
speaker, `Speaker 1`, plus a `warn` event.

---

## 8. Summarizer contract

### 8.1 Invocation
```
<prompt on stdin> | /home/USER/.local/bin/claude -p \
    --model sonnet --output-format json --tools "" --safe-mode --no-session-persistence
```
`cwd = data/tmp/claude`, `timeout = summary.timeout_s`, environment copied from the worker with
`ANTHROPIC_API_KEY` **removed** so the subscription OAuth session is used and nothing is billed per
call. `--safe-mode` disables CLAUDE.md, skills, hooks and MCP for determinism; `--tools ""` is the
documented "disable all tools" form; `--no-session-persistence` keeps `~/.claude` from filling up.
*(All flags verified against Claude Code 2.1.269 on this VPS.)*

Envelope (verified): one JSON object with `"type":"result"`, `"subtype":"success"`,
`"is_error":false`, `"api_error_status":null`, `"result":"<model text>"`. Treat as failure: non-zero
exit, `is_error != false`, `subtype != "success"`, non-null `api_error_status`, or missing `result`.
Parse `result` after stripping an optional ```` ```json ```` fence.

### 8.2 Required output schema
The prompt demands **only** this object, no prose:
```json
{"titulo":"string (<=80 chars, no date, no quotes) | null",
 "resumen_corto":"1-2 sentences","resumen":"one paragraph",
 "mas_detalle":[{"tema":"string","texto":"string"}],
 "puntos_clave":[{"emoji":"single emoji","tema":"string","texto":"string"}],
 "tareas":[{"responsable":"string","texto":"string","t_seg":128}],
 "temas":["string"],"palabras_clave":["string"],
 "categoria":"one of the configured category keys","idioma":"es | en",
 "cliente_sugerido":"one of the configured client keys | null","confianza":0.0}
```
Rules stated in the prompt: output language = the language of the recording; `titulo` only when the
user gave none (else `null`); 4–8 `puntos_clave`; `mas_detalle` mirrors `puntos_clave`; `t_seg` is
an integer second offset taken from the `[MM:SS]` marks; `responsable` is a role or a `Speaker N`,
never an invented name; 4–8 `palabras_clave`; `categoria` and `cliente_sugerido` must come from the
lists injected into the prompt.

### 8.3 Chunking
`chunk_turns` accumulates whole turns until a chunk reaches `summary.chunk_seconds` (1500 s), no
overlap, never splitting a turn. Prompt lines are `[MM:SS] Speaker 1: texto`. One chunk → a
**single call** returning the full object (no reduce). N > 1 → N **map** calls each returning only
`{resumen_corto, puntos_clave, tareas, temas, palabras_clave}` for that chunk, then one **reduce**
call that receives the N map objects plus the hints — never the raw transcript, so the reduce stays
small regardless of meeting length.

### 8.4 Validation and failure policy
`validate(obj) -> SummaryResult` coerces: missing optional keys → empty list / `null`; wrong-typed
elements dropped; `confianza` clamped to `[0,1]`; `idioma` outside `idiomas_permitidos` → the ASR
language; unknown `categoria` → `"otras"`; `cliente_sugerido` not a configured key → `null`;
`t_seg` clamped to `[0, duration_s]`. Unparseable output, or both `resumen_corto` and
`puntos_clave` empty, counts as failure.

On failure: **one** retry with the same prompt prefixed by `Tu respuesta anterior no era JSON
válido. Responde ÚNICAMENTE con el objeto JSON.` If that also fails, `summarize` raises
`SummaryError`; the worker writes `summary.json` with `{"schema":1,"pending":true,
"error":"<reason>", …empty fields…}`, renders the note with the `Resumen pendiente` marker (§9.4),
**delivers normally**, and sets `summary_pending`. In a map-reduce run a failing *map* chunk is
skipped with a `warn` event as long as one chunk succeeded; only a failing *reduce* makes the whole
summary pending.

---

## 9. Output renderer

### 9.1 Client assignment
First hit wins, recorded in `client_source`: **(1)** explicit `client` upload field (must be a
configured key); **(2)** case-insensitive, accent-folded substring match of any `marcadores` entry
against `titulo + " " + " ".join(palabras_clave)`; **(3)** `summary.cliente_sugerido` when
`confianza >= summary.confianza_minima_cliente`; **(4)** `sin_clasificar` → `_Sin_clasificar`.
**A client folder is never created for a key absent from `config.toml`.**

### 9.2 Start time
`recorded_at` upload field → `ffprobe` `creation_time` → filename patterns → file mtime. Patterns,
in order, against the original basename: `YYYYMMDD HHMMSS`, `YYYY-MM-DD HH.MM.SS`,
`YYYY-MM-DD_HH-MM-SS`, `YYYYMMDD_HHMMSS`. Everything converted to `Europe/Madrid`.

### 9.3 Fields
`titulo` = user title → `summary.titulo` → `"Reunión sin título"`, through `sanitize_title`.
`duracion_min = max(1, round(duration_s / 60))`. `enlace` is always `null`. `participantes` =
`turns.participantes`. `hash_origen` = `audio_sha256`. `generado_en` = now, ISO with offset.
`hash_contenido` = SHA-256 of the rendered **body** (everything after the front-matter's closing
`---` and the blank line following it), UTF-8.

### 9.4 Template
Front-matter: `---`, then exactly these keys in this order, one per line, value JSON-encoded
(`json.dumps(v, ensure_ascii=False)`), then `---` and one blank line:
`generador, tipo, origen_id, titulo, fecha, hora, inicio, duracion_min, idioma, categoria,
palabras_clave, participantes, enlace, resumen_corto, cliente, cliente_nombre, hash_origen,
version_modulo, generado_en, hash_contenido`.

Body, in order, headings exactly as written:
* `# <titulo>`
* blockquote, each line ending in two spaces: `> **<fecha larga>** · HH:MM · N min · <Categoría display>`
  and `> Palabras clave: a, b, c`
* when the summary is pending, a third line:
  `> ⚠️ **Resumen pendiente** — la transcripción está completa; el resumen se generará más tarde.`
* `## Resumen` — the paragraph, then `**Más detalle:**` and one blank-line-separated bullet per
  `mas_detalle` entry: `- **<tema>:** <texto>`
* `## Puntos clave` — `- <emoji> <tema>: <texto>`
* `## Tareas y acuerdos` — grouped by `responsable` in order of first appearance; each group
  `### <responsable>` then `- [ ] <texto> · [MM:SS]`
* `## Temas tratados` — `- <tema>`; when empty the literal `_Sin temas listados._`
* `## Transcripción completa` — per turn `**Speaker 1** · [MM:SS]`, newline, text, blank line

Empty `Resumen` / `Puntos clave` / `Tareas y acuerdos` sections use the same `_Sin … listados._`
placeholder style, so the section set is invariant.

### 9.5 Full example

```markdown
---
generador: "transcription-24-7"
tipo: "reunion"
origen_id: "01k5r2m8q7v3x9b0c1d2e3f4g5"
titulo: "Entrega de libros tutora"
fecha: "2026-09-19"
hora: "12:35"
inicio: "2026-09-19T12:35:11+02:00"
duracion_min: 4
idioma: "es"
categoria: "tutoria"
palabras_clave: ["reparto de libros","tutores","código interno","hoja de observaciones","grupos"]
participantes: ["Speaker 1","Speaker 2"]
enlace: null
resumen_corto: "Se detallan las funciones de los tutores el día del reparto de libros de gratuidad."
cliente: "centro_ejemplo"
cliente_nombre: "Centro Ejemplo"
hash_origen: "87aa1a29fc25fdcb13b368e26d2f7032877a89b0c2973f77d2521be7d9dfeb41"
version_modulo: "1.0.0"
generado_en: "2026-09-19T12:41:02+02:00"
hash_contenido: "40555bf609e232af3c56d40b84d02a402e3de3ddca760a72b15d884b19049441"
---

# Entrega de libros tutora

> **sábado 19 de septiembre de 2026** · 12:35 · 4 min · Tutoría  
> Palabras clave: reparto de libros, tutores, código interno, hoja de observaciones, grupos  

## Resumen

Se detallan las funciones de los tutores y tutoras el día del reparto de libros de gratuidad.

**Más detalle:**

- **Organización del reparto:** El tutor acude al patio 10 minutos antes de su franja.

- **Documentación:** Cada participante cumplimenta código interno y hoja de observaciones.

## Puntos clave

- 📚 Reparto de libros: El grupo se ordena alfabéticamente y entra por turnos al gimnasio.
- 📝 Documentación: Se rellena una hoja de observaciones por asignatura.

## Tareas y acuerdos

### Tutores
- [ ] Estar en el patio 10 minutos antes de la franja asignada · [00:12]
- [ ] Entregar en secretaría los lotes no recogidos · [02:01]

## Temas tratados

- Reparto de libros de gratuidad
- Cumplimentación del código interno

## Transcripción completa

**Speaker 1** · [00:01]
Hola. En septiembre llevaremos a cabo el reparto de libros de gratuidad.

**Speaker 2** · [02:01]
Finalmente, el nombre y apellidos del participante.
```

### 9.6 Atomic, non-destructive writes
`write_note(path: Path, content: str) -> Path`:
1. `guard_write(path)`; `mkdir(parents=True, exist_ok=True)` on the parent.
2. If the target exists, parse its front-matter, recompute the SHA-256 of its body and compare with
   its own `hash_contenido`. **Mismatch → the user edited it**: write `<stem> (2).md` (then `(3)`, …)
   and log a `warn`. Match → overwrite in place.
3. Write `<path>.tmp-<pid>` in the same directory, `flush` + `os.fsync`, then `os.replace`.
4. Return the path actually written; store it in `jobs.note_path`.

Re-running a delivered job is idempotent: identical inputs yield an identical body and therefore an
identical `hash_contenido`, so the same file is rewritten with the same bytes apart from `generado_en`.

---

## 10. Indices

Regenerated **in full** after every delivery and by `t247 reindex`, from a filesystem scan of
`<Cliente>/AAAA-MM/*.md` front-matter (authoritative, so a hand-deleted note disappears).
Front-matter only — never the transcript body.

```json
{"version":"1.0.0","generado_en":"2026-09-19T12:41:03+02:00","reuniones":[
 {"id":"01k5…","fecha":"2026-09-19","hora":"12:35","inicio":"2026-09-19T12:35:11+02:00",
  "titulo":"Entrega de libros tutora","cliente":"centro_ejemplo",
  "cliente_nombre":"Centro Ejemplo","categoria":"tutoria",
  "palabras_clave":["reparto de libros"],"duracion_min":4,"idioma":"es","resumen_corto":"…",
  "participantes":["Speaker 1","Speaker 2"],"resumen_pendiente":false,
  "ruta":"Centro_Ejemplo/2026-09/2026-09-19 12h35 - Entrega de libros tutora.md"}]}
```

Sorted by `inicio` descending; `ruta` is POSIX-relative to `archivo.raiz`, **unescaped**.

`INDICE.md` (root): front-matter `generador`, `tipo: "indice_archivo"`, `reuniones: <n>`,
`version_modulo`; an H1; a generated-automatically line linking `LEEME.md` and the JSON catalog; a
`**Por cliente:**` line with per-client links and counts; then the table
`| Fecha | Hora | Cliente | Reunión | Categoría | Palabras clave | Duración |`, newest first.
`<Cliente>/INDICE.md` is the same without the Cliente column, `tipo: "indice_cliente"`, plus
`cliente`/`cliente_nombre`, paths relative to the client folder.

**`Tareas abiertas del owner` (parity with the Fireflies archive).** Every `<Cliente>/INDICE.md`
ends with a `## Tareas abiertas del owner` section built from the notes of that client only. A
note's `## Tareas y acuerdos` section is walked; a `### <responsable>` group counts as the owner's when
the heading contains any marker from `archivo.responsables_owner` (default `["Owner", "(tú)"]`),
matched case-insensitively and accent-folded as a substring — so `### Owner`, `### owner` and
`### Coordinación TIC (tú)` all qualify and `### Tutores` does not. Only **unchecked** `- [ ]`
items are collected (`- [x]` is done and disappears on the next rebuild, which is what makes the
index a live to-do list); they are grouped by meeting, newest first, under a bold link to the note
(client-relative, percent-encoded like every other link here), and re-emitted as `- [ ]`
checkboxes. Empty → the literal `_Sin tareas abiertas._`, so the section set stays invariant as in
§9.4. The root `INDICE.md` does not carry the section. To make the input exist, the summarizer
prompt (§8.2, `tareas_rule`) instructs the model to suffix `responsable` with `(tú)` whenever the
transcript makes clear the *recorder* took the task on ("lo hago yo", "me encargo").

Link targets are **percent-encoded** (`urllib.parse.quote(ruta, safe="/")`), matching the existing
archive; `|` inside a cell is escaped `\|`; `Duración` renders `N min`. A client `INDICE.md` exists
only while that client has ≥1 note, and is removed when it drops to zero. `LEEME.md` is written when
absent or when `version_modulo` changes: structure tree, client table, front-matter key list,
section list, catalog schema, and a note that speakers are numbered by first appearance and not
named. All four files go through `guard_write`.

---

## 11. systemd (user units)

Linger is already enabled. Units install to `~/.config/systemd/user/`; `%h` is the home directory.

**Escaping (verified on systemd 257, this VPS).** The two settings that carry the project path need
*different* treatment, and getting it wrong yields a unit that verifies cleanly and then fails at
runtime: `ExecStart=` is parsed into an argv, so each space must be `\x20`; `WorkingDirectory=`
takes the rest of the line as one path, so the spaces are written **literally** — `\x20` there is
not unescaped (the daemon ends up with a directory whose name contains `\x20`) and quoting the
value yields an empty `WorkingDirectory`. Both files are checked with
`systemd-analyze --user verify` by `ops/install.sh`.

`ops/t247-worker.service`
```ini
[Unit]
Description=Transcription 24/7 worker (ASR + diarization + summary)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/transcription-stack-template
ExecStart=%h/transcription-stack-template/.venv/bin/t247 worker
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=4
Environment=TZ=Europe/Madrid
Restart=on-failure
RestartSec=10
RestartSteps=5
RestartMaxDelaySec=300
Nice=19
CPUSchedulingPolicy=idle
IOSchedulingClass=idle
IOSchedulingPriority=7
OOMScoreAdjust=500
CPUQuota=400%
MemoryHigh=2500M
MemoryMax=3G
MemorySwapMax=0
KillSignal=SIGTERM
TimeoutStopSec=120
StandardOutput=journal
StandardError=journal
SyslogIdentifier=t247-worker

[Install]
WantedBy=default.target
```

`ops/t247-api.service`
```ini
[Unit]
Description=Transcription 24/7 upload API (tailnet only)
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=%h/transcription-stack-template
ExecStart=%h/transcription-stack-template/.venv/bin/t247 serve
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=TZ=Europe/Madrid
Restart=on-failure
RestartSec=5
RestartSteps=4
RestartMaxDelaySec=60
Nice=19
CPUSchedulingPolicy=idle
IOSchedulingClass=idle
OOMScoreAdjust=500
MemoryHigh=250M
MemoryMax=300M
MemorySwapMax=0
TimeoutStopSec=20
StandardOutput=journal
StandardError=journal
SyslogIdentifier=t247-api

[Install]
WantedBy=default.target
```

**Tailscale ordering.** `tailscaled` is a *system* unit, so a `--user` unit cannot order itself
after it: no `After=`/`Wants=` combination works across that boundary. Instead `t247 serve` fails
fast when `api.host` is not yet on any local interface (§12), `Restart=on-failure` brings it back,
and `StartLimitIntervalSec=0` disables systemd's start rate limiting so the retry loop survives a
slow tailnet however long it takes. Both units restart **on failure only**, with a growing delay
(`RestartSteps`/`RestartMaxDelaySec`, systemd ≥ 254), so a crash loop never becomes a busy loop on
a shared box — and a clean `systemctl --user stop` stays stopped.

Units call `.venv/bin/t247` directly rather than `uv run`, so a restart never touches the network.

`ops/install.sh` (bash, `set -euo pipefail`, absolute paths, idempotent): **(1)** resolve `PROJECT`
from the script location, verify `.venv/bin/t247`; **(2)** `mkdir -p ~/.config/transcription247 &&
chmod 700`, create `api_token` with `openssl rand -hex 32` + `chmod 600` if missing — **never
printing it**, only the path to read it from; **(3)** create `data/{jobs,tmp,logs,tmp/claude}` and
the Drive root + `_INBOX_AUDIO/{_procesados,_error}`, each `mkdir` going through
`paths.guard_write()` so the installer itself cannot create a directory outside the owned tree; **(4)** `install -m 644` both units; **(5)** `systemctl --user daemon-reload && systemctl
--user enable --now t247-worker.service t247-api.service`; **(6)** run `t247 doctor`, exit non-zero
on failure. No onnxruntime soname step: §16 D9 (updated) — the pinned `sherpa-onnx-core==1.13.8`
dependency already installs a matching `libonnxruntime.so` inside `site-packages/sherpa_onnx/lib/`,
so `import sherpa_onnx` works straight out of `uv sync` with no install-time fixup.

`ops/uninstall.sh`: `systemctl --user disable --now` both units, remove the unit files,
`daemon-reload`, leave `data/` and the Drive output untouched and print where they are.

---

## 12. Security

* **Bind address.** `api.host` defaults to the tailnet IP `<TAILSCALE_IP>`. `api.py` refuses to
  start — hard `SystemExit`, not a warning — if the host is `0.0.0.0`, `::`, or any address not
  present on a local interface, and `config.py` rejects a wildcard host at load time as well.
  Tailnet-only is a **requirement, not a default**: there is deliberately no override flag
  (`ApiConfig` has no `allow_public` field and must not grow one). No reverse proxy, no port
  forwarding.
* **Token.** 32 random bytes hex in `~/.config/transcription247/api_token` (0600 inside a 0700
  directory); refused if the mode is wider; compared with `hmac.compare_digest`; never logged, and
  request logging redacts the `Authorization` header.
* **Path traversal.** Every write goes through `guard_write` (§3.1). Uploaded filenames are reduced
  to `Path(name).name` then sanitized, so `..`, absolute paths, NUL and control characters cannot
  survive. Client folders come only from `config.toml` — never from user input or from Claude.
* **No writes outside the owned tree.** `guard_write` hard-denies any path containing
  `_Protected_Archive` (a protected sibling archive) or `_Protected_Private`. Enforced in code, not by
  convention, and covered by a unit test.
* **Upload size.** Enforced while streaming (`api.max_upload_bytes`, 1 GiB) and again by
  `MemoryMax=300M`, which a non-streaming implementation would trip immediately.
* **Secrets.** None in the repo; `data/` is gitignored; `config.toml` holds no credentials. The
  summarizer scrubs `ANTHROPIC_API_KEY` from the child environment so the subscription session is
  used and no per-call charge can occur.
* **Subprocesses.** `ffmpeg`, `ffprobe` and `claude` are invoked as argument lists, never via a
  shell; `--tools ""` means the summarizer model cannot touch the filesystem.
* **Log hygiene.** Transcript text is never logged; events carry counts, durations and stage names.

---

## 13. Failure modes

| # | Condition | Detection | Response |
|---|---|---|---|
| 1 | Drive mount down / `archivo.raiz` missing | `is_dir()` pre-flight; `OSError` on write | Job stays at its last state with `next_attempt_at` set; `/healthz` → 503; notify at most hourly, not per retry |
| 2 | `_INBOX_AUDIO` unreadable | poller `OSError` | Poller logs `warn`, backs off to 60 s, retries; never crashes the worker |
| 3 | `claude` not logged in / rate-limited | non-zero exit, `is_error`, `api_error_status` | One retry, then deliver with `Resumen pendiente`, state `summary_pending`, notify |
| 4 | `claude` returns prose or broken JSON | `json.loads` / validation fails | One retry with the corrective preamble, then as #3 |
| 5 | `MemAvailable` < `worker.mem_disponible_minima_mb` | `/proc/meminfo` before each job **and** before the ASR and diarization stages | Sleep `mem_espera_segundos`, re-check up to `mem_max_esperas`; then leave the job queued and notify |
| 6 | Corrupt / unsupported audio | ffmpeg non-zero, or duration < `min_duracion_s` | `failed` immediately, no retries; inbox file → `_error/` + `.txt` reason; notify |
| 7 | 0 speech detected (`words == []`) | after ASR | `failed`, `error = "no_speech"`; no note written; inbox file → `_error/`; notify |
| 8 | Duplicate upload (same SHA-256) | `UNIQUE(audio_sha256)` | API 200 `duplicate:true` with the existing job; inbox path: never enqueues a second job. A file whose *own path* already has a non-terminal job (`Store.find_active_by_origin_ref`) is left untouched and skipped without re-hashing — this also covers the same original reappearing on a later poll while its job is still running. A genuinely *new* path whose SHA-256 matches a job that is `delivered`/`summary_pending` is a true duplicate: removed/archived per `inbox.after_success`. One matching a `failed` job is moved to `_error/` (it can never be enqueued while that row exists, and would fail the same way). One matching a job that is still in flight (different path, same content) is left alone until that job goes terminal, then re-evaluated on a later poll |
| 9 | Disk full | `shutil.disk_usage` pre-flight (2× upload size); `ENOSPC` | API 507; worker leaves the job queued, prunes artifacts older than `conservar_artefactos_dias`, notifies |
| 10 | Worker killed mid-stage (OOM, restart) | stale `lease_until` at startup | Requeue; completed artifacts are reused, so work restarts at the last finished stage |
| 11 | Note edited by hand, then re-run | body hash ≠ stored `hash_contenido` | Write a `(2)` sibling; never overwrite the edit |
| 12 | Drive file still uploading | size changed between polls | Require two equal readings ≥ `estabilidad_segundos` apart before copying |
| 13 | Two workers started by accident | claim `UPDATE … WHERE lease_until IS NULL` | Second worker claims nothing and idles; systemd already prevents a second instance |

### 13.1 `docs/CLIENTES_IPHONE_MAC.md` — required contents (written in [INT])
**iPhone primary:** a Shortcut *Transcribir* accepting audio from the Share Sheet, using **Get
Contents of URL** → `POST http://<TAILSCALE_IP>:8247/upload`, Request Body `Form`, file field
`file`, optional `title`, header `Authorization: Bearer …` with the token in a Text action; how to
add it to the Voice Memos share sheet; how to confirm Tailscale is connected.
**iPhone fallback:** Google Drive app → *Save to Drive* → `…/Transcripciones_24_7/_INBOX_AUDIO`
(works with Tailscale off). **Mac primary:** Google Drive for desktop, drop into the synced
`_INBOX_AUDIO`. **Mac alternative:** a `launchd` `WatchPaths` plist calling a `curl` one-liner with
the token in a 0600 file. **Optional:** capturing meeting system audio on the Mac with a
ScreenCaptureKit CLI recorder or BlackHole + an aggregate device. Plus: checking status at
`http://<TAILSCALE_IP>:8247/`, and the expected turnaround.

---

## 14. Work breakdown

Five parallel packages with **zero file overlap**, then integration. Each codes against the
signatures and schemas above and needs no conversation with the others.

**P1 — Core: config, paths, store, ids, logging, CLI skeleton · standard.**
Owns `config.toml` and `src/transcription247/{__init__,cli,config,paths,ids,logging_setup,store,models}.py`.
Depends on nothing. `cli.py` defines every subcommand (`worker serve enqueue status retry
resummarize reindex doctor`) and imports handlers by their documented paths — `worker.run_worker`,
`api.serve`, `worker.enqueue_file`, `worker.print_status`, `worker.retry`, `worker.resummarize`,
`indices.rebuild_all`, `doctor.run` — so **no other package edits `cli.py`**. `enqueue` takes a file
path plus the same optional hints as `POST /upload` (§6): `--title`, `--language`, `--client`,
`--recorded-at`, and `--speakers N` (`1`-`10`) — `worker.enqueue_file(path, ..., speakers:
int | None = None)` stores it as `jobs.speakers_user` exactly like the upload form field (the
filename hint, §6, applies here too when `--speakers` is omitted). P1 lands first; the others code
against §3–§5 meanwhile.

**P2 — Pipeline: audio, ASR, diarization, merge · complex.**
Owns `{audio,asr,diarize,merge}.py`. Depends on `models.py` + `config.py` dataclasses. Riskiest
package: model lifecycle (`__enter__`/`__exit__` must genuinely free memory) and the merge
algorithm. Developable against hand-written `asr.json`/`diarization.json` fixtures with no model.

**P3 — Summarizer · standard.**
Owns `{summarize,prompts}.py`. Depends on `models.py` + `config.py`. Testable end to end with a fake
`claude_bin` that echoes a canned envelope. Must implement §8 exactly, including the `pending` path.

**P4 — Renderer and indices · standard.**
Owns `{render,indices}.py`. Depends on `paths.py` (`guard_write`, `fmt_ts`, `fmt_long_date_es`,
`note_path`) + `models.py`. Works entirely from `turns.json` + `summary.json` fixtures; the
golden-file test (§15) is its acceptance gate.

**P5 — Ingestion: HTTP API, web page, inbox poller, notifier · standard.**
Owns `{api,web,inbox,notify}.py`. Depends on `store.py`, `config.py`, `paths.py`. Never runs
pipeline stages — it only writes rows and files. The poller is exposed as
`inbox.start_poller(cfg, store) -> threading.Thread` (daemon) so the worker starts it; the API
process never does.

**INT — Integration and verification · complex.**
Owns `{worker,doctor}.py`, `ops/*`, `tests/*`, `README.md`, `docs/CLIENTES_IPHONE_MAC.md`. Wires the
stage sequence, pre-flight RAM checks, leases, resume, retries, artifact pruning, delivery and
notifications; writes the units and installer; owns all tests. Starts once P1 merges, lands last.

**Inbox poller placement:** a daemon thread inside the worker process, not a systemd timer. A timer
would spawn a fresh Python interpreter every 15 s (~40 MB of allocation churn, 5 760 starts a day)
on a RAM-constrained box for work that is a single `os.scandir`. The thread only writes queue rows,
so the serial single-job guarantee is unaffected.

---

## 15. Test plan

Four high-value `pytest` tests, no models and no network, target < 5 s total.

1. **`test_merge.py`** — synthetic words/segments. Max-overlap picks the right speaker for a word
   straddling a boundary; a 1-word 0.3 s flip between two identical neighbours is absorbed; a 1.5 s
   silence starts a new turn; renumbering follows first appearance (raw cluster `2` speaking first
   yields `Speaker 1`); empty diarization yields one speaker; empty words raises.
2. **`test_render_golden.py`** — `fixtures/{turns,summary}.json` → byte-for-byte comparison with
   `fixtures/expected_note.md`, with `generado_en`/`hash_contenido` normalized. Second case: the
   `pending` variant. Third: re-rendering an edited file produces a `(2)` sibling and leaves the
   original bytes untouched.
3. **`test_paths.py`** — `guard_write` rejects `../`, absolute escapes, symlink escapes and any path
   under the sibling `_Protected_Archive/` or `_Protected_Private/`, and accepts legitimate targets under
   all three roots. `sanitize_title` handles `/`, NUL, control chars, leading dots, 300-char input
   and `""`. `fmt_ts(59.4) == "00:59"`, `fmt_ts(3600) == "1:00:00"`.
4. **`test_store_resume.py`** — tmp DB: a job at `transcribed` with a stale lease is requeued at
   startup and resumes at `diarized`; a valid existing artifact makes its stage skip; three failures
   reach `failed` with the documented backoffs; a duplicate SHA-256 insert returns the existing job.

Explicitly **not** tested: faster-whisper, sherpa-onnx, real `claude` calls, real Drive I/O — those
are covered by `t247 doctor` and by `bench/`.

---

## 16. Decisions

* **D1** Job ids are 26-char lowercase Crockford ULIDs (stdlib-generated): time-sortable, opaque,
  and a drop-in for the old `fireflies_id` slot as `origen_id`.
* **D2** FastAPI + uvicorn over stdlib `http.server` (justified in §6).
* **D3** The inbox poller is a daemon thread inside the worker, not a systemd timer (§14).
* **D4** `summary_pending` is a state, not a flag: *delivered without a summary*. Such notes appear
  in the indices with `resumen_pendiente: true`.
* **D5** `participantes_email` is dropped from the front-matter (no e-mails to record) and `enlace`
  is always `null`; every other Fireflies key is kept, in the same order.
* **D6** `hash_contenido` covers the rendered body only, making hand-edit detection self-contained —
  no external bookkeeping of what we last wrote.
* **D7** Category keys are a closed, configured set with display labels; anything Claude returns
  outside it becomes `otras`.
* **D8** Indices are always regenerated in full from a filesystem scan (tens of files, milliseconds)
  rather than patched incrementally, so they self-heal.
* **D9** *(verified on this VPS, 2026-09-19 — fixed)* `sherpa-onnx` 1.13.8's extension declares
  `NEEDED libonnxruntime.so` with version tag `VERS_1.28.2`, while a bare `onnxruntime` dependency
  resolves to the latest release (1.30.0), which ships only `libonnxruntime.so.1.30.0` — an
  immediate `ImportError` on `import sherpa_onnx`. **Fix (applied, no owner action left):** pin
  `sherpa-onnx-core==1.13.8` explicitly in `pyproject.toml` (`uv add sherpa-onnx-core==1.13.8`) —
  that package vendors the exact matching `libonnxruntime.so` directly inside
  `site-packages/sherpa_onnx/lib/` (already on the extension's RPATH), so no `onnxruntime==1.28.2`
  pin and no `ops/install.sh`/`doctor` symlink step are needed. `pyproject.toml`/`uv.lock` already
  carry the pin; `t247 doctor` still checks `import sherpa_onnx` explicitly as a regression guard.
* **D10** *(verified)* Claude Code 2.1.269 accepts `-p --model sonnet --output-format json
  --tools "" --safe-mode`; `--tools ""` is the documented "disable all tools" form; the envelope
  carries the model text in `result` as a string. `--bare` must **never** be used — it forces
  `ANTHROPIC_API_KEY` auth and would bill per call instead of using the subscription.
* **D11** systemd units call `.venv/bin/t247` directly instead of `uv run`, so a restart never needs
  the network.
* **D12** Speaker renaming is a config hook (`archivo.renombrar_speakers`, default `false`) and
  nothing more; labels stay `Speaker N`.
* **D13** `inbox.after_success` defaults to `delete`; `_procesados/` is created anyway so switching
  to `archive` needs no migration.
* **D14** *(measured, `bench/RESULTS.md`, 2026-09-19)* ASR (large-v3-turbo int8, beam=5,
  cpu_threads=4): RTF ≈ 0.27-0.43, peak RSS ≈ 1.66 GB. Diarization (sherpa-onnx, wespeaker
  embedding, `num_speakers` known): RTF ≈ 0.09-0.15, peak RSS ≤ 440 MB. Run strictly sequentially
  (§1), a 1 h recording takes ≈ 21.5 min end to end (ASR + diarization only; summary and render add
  little). These numbers back the `# bench` defaults in §5 and the `MemoryMax` budgets in §11.
* **D15** `speakers` (upload field / `--speakers` / filename hint `[Np]`/`(Np)`, §6, §14) is
  optional and additive to the existing pipeline: when absent, behavior is unchanged (auto
  clustering, §3.5). When present it is always preferred because a known count is cheap and
  reliably correct (D14's bench data), never because the auto path is assumed wrong for that job.
* **D17** *(measured, `bench/RESULTS.md` "Auto speaker count", 2026-09-19)* With an unknown speaker
  count, a single cosine threshold on sherpa's per-window embeddings is not tunable to a safe
  default — on an 8-clip dev set the shipped `cluster_threshold = 0.90` got 3/8 counts right and
  collapsed a 4-speaker panel to 1. Replaced by `auto_count = "overcluster"` (§3.5): over-split,
  re-embed each over-cluster with a **multilingual** embedding (3D-Speaker eres2net) instead of the
  VoxCeleb-English one, pool per over-cluster, then merge the centroids (cosine, average linkage,
  cut at 0.475) and absorb clusters under 3 % of the speech. 7/8 exact, and the two 1-speaker clips
  stay at 1 across the whole 0.20-0.65 threshold range. The English VoxCeleb embedding was the main
  culprit: on Spanish audio its pooled centroids for *different* speakers sit 0.22 apart while a
  single speaker's own centroids reach 0.21 — no threshold can separate those. Cost: diarization
  RTF ≈ +30 % over the known-count path (0.09 → ~0.12 on an idle box) and **peak RSS 250 MB →
  0.46-1.11 GB** depending on the clip, from holding a second onnxruntime session; still under
  ASR's 1.66 GB, so the §11 `MemoryMax` budgets and `worker.mem_disponible_minima_mb` are
  unchanged — ~~but no clip in the dev set is longer than 6 min, so memory on a 1-2 h recording is
  untested~~ **(that caveat was the real defect: long recordings OOMed. Fixed and measured in
  D18.)** The known-count path is untouched and still preferred whenever the caller knows the
  count. Auto counting is reliable for 1-4 speakers; beyond that it under-counts (a 5-speaker
  far-field panel came out as 2).
* **D18** *(measured, `bench/RESULTS.md` "Long audio", 2026-09-19)* **Peak RSS used to grow with
  the recording's duration, so real jobs could not be processed at all.** A 60-minute wav was
  OOM-killed 17 s into the ASR — `WhisperModel.transcribe()` runs its VAD eagerly over the whole
  array (≈10 MB per audio-minute) before decoding anything — and diarization's `process()` held
  ≈54 MB per audio-minute, extrapolating to ≈6.6 GB at 2 h. 40 min also OOMed; 30 min survived at
  2459 MB, already over `MemoryHigh`. Every earlier memory figure in this document was taken on
  clips of ≤6 min and silently assumed duration-independence, which was wrong.
  **Fix:** a shared `windows.py` splits the normalized wav into contiguous windows of at most
  `pipeline.window_max_s` (900 s), cut at the longest low-energy pause in the window's last 90 s
  (hard cut when the tail is unbroken speech), and both heavy stages process one window at a time,
  reading its samples from disk on demand. ASR offsets every timestamp and pins the language from
  the first window with speech; diarization over-clusters per window, pools one centroid per
  `(window, local cluster)`, and takes **one global agglomerative decision** over all of them, so
  speaker ids are consistent end to end. Only centroids cross a window boundary.
  **Measured:** peak RSS **1857 MB at 20 min → 1903 MB at 60 min → 1909 MB at 120 min**, i.e.
  **flat** in the duration (+52 MB over a 6× range, and 60→120 min costs 6 MB); the residue is the
  accumulated word/segment lists, not the models. ~590 MB of headroom under `MemoryHigh` at any
  length. RTF is unchanged and linear (ASR 0.176-0.191, ≈0.34 end to end, ~20 min of processing per
  hour of audio). Speaker count stayed correct (**2**) at all three durations: at 120 min, 127
  window-local clusters collapsed to 2 global speakers, both appearing on both sides of all eight
  cuts.
  **Cost / trade-offs:** windowing is *not* free at the joins — a turn straddling a cut is
  transcribed as two utterances and re-glued by `_merge_adjacent`, and a speaker who appears only
  within one window has less material to link on. Both are why the cut hunts for a pause instead of
  slicing at a fixed offset. Choosing 900 s is a memory/quality trade: a smaller window lowers the
  peak roughly linearly and adds joins. Single-window recordings (< 15 min, i.e. most clips in the
  dev set) take the original code path verbatim — `bench/diar_e2e.py` reproduces all 24 baseline
  results (8 clips × 3 modes) with **zero differences**, so this bought long-audio capability
  without moving short-audio behaviour at all.
* **D16** *(fixed, 2026-09-19)* The inbox poller used to delete/archive a job's Drive original as
  soon as *any* file with a matching SHA-256 was seen — including the job's own original
  reappearing on a later poll while it was still in flight, since it is never removed before
  delivery. `inbox.after_success`/`after_failure` would then find nothing left to act on. Fixed:
  `Store.find_active_by_origin_ref(origin_ref)` lets the poller ask the DB (not the poller's
  in-memory `tracker`, so this survives a worker restart) whether a candidate's own path already
  has a non-terminal job before touching it at all, and a duplicate SHA-256 under a *different*
  path is only ever acted on once the job it matches is terminal (§13 #8).
