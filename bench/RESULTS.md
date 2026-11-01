# Benchmark results — faster-whisper + sherpa-onnx (CPU only)

Run 2026-09-19 on the shared Debian 13 VPS (8 vCPU AMD EPYC, 16 GB RAM), each stage run
under `systemd-run --user` with `CPUQuota=400% MemoryMax=3G MemorySwapMax=0 Nice=19
CPUSchedulingPolicy=idle IOSchedulingClass=idle OOMScoreAdjust=500`. ASR and diarization
ran as **separate subprocesses** (see `bench/bench.py`), so the peak RSS numbers below are
per-stage, not cumulative — the pipeline's true peak at any instant is `max(stage)`, here
≈1.66 GB (ASR), comfortably inside the 3 GB cap.

Audio: `bench/audio/es_entrevista_5min.wav` (5:00, Spanish, Bolivian interview, 2 speakers)
and `bench/audio/en_interview_2min.wav` (2:00, English, interview, 2 speakers). Sources and
licenses in `bench/README.md`.

## ASR — faster-whisper large-v3-turbo, int8, cpu_threads=4

| Clip | beam | process time | RTF (process-only) | RTF (incl. model load) | peak RSS | language detected |
|---|---|---|---|---|---|---|
| ES 5min | 5 | 81.0 s | 0.270 | 0.280 | 1662 MB | es (p=0.9987) |
| ES 5min | 1 | 60.9 s | 0.203 | 0.211 | 1662 MB | es (p=0.9987) |
| EN 2min | 5 | 51.7 s | 0.431 | 0.456 | 1663 MB | en (p=0.99982) |
| EN 2min | 1 | 23.1 s | 0.192 | 0.215 | 1662 MB | en (p=0.99982) |

Model load time itself is small and constant (~2.4–3.0 s) regardless of clip length, so it
matters only for very short/streaming-style jobs, not for meeting-length audio. Language
auto-detect was correct and highly confident on both clips.

beam=1 is ~1.3–2.2× faster than beam=5 for a word-error-rate cost that (qualitatively, see
below) is not obviously visible on these clips — see recommendation.

## Diarization — sherpa-onnx, pyannote-segmentation-3.0 + speaker embedding

Actual speaker count on both clips: **2** (interviewer + interviewee).

| Clip | embedding | mode | process time | RTF | peak RSS | speakers found |
|---|---|---|---|---|---|---|
| ES 5min | eres2net (3D-Speaker, zh-cn) | auto (threshold=0.5) | 43.0 s | 0.143 | 442 MB | **18** ❌ |
| ES 5min | eres2net | num_speakers=2 | 41.5 s | 0.138 | 413 MB | 2 ✓ |
| ES 5min | wespeaker (VoxCeleb, en, resnet34) | auto (threshold=0.5) | 28.6 s | 0.095 | 241 MB | 2 ✓ (lucky, see below) |
| ES 5min | wespeaker | num_speakers=2 | 26.4 s | 0.088 | 255 MB | 2 ✓ |
| EN 2min | eres2net | auto (threshold=0.5) | 16.5 s | 0.138 | 389 MB | **13** ❌ |
| EN 2min | eres2net | num_speakers=2 | 17.6 s | 0.147 | 386 MB | 2 ✓ |
| EN 2min | wespeaker | auto (threshold=0.5) | 12.2 s | 0.101 | 223 MB | **8** ❌ |
| EN 2min | wespeaker | num_speakers=2 | 12.2 s | 0.102 | 222 MB | 2 ✓ |

**The default clustering threshold (0.5) for "unknown speaker count" is clearly poor** — it
over-segmented into 8–18 pseudo-speakers on 3 of 4 clip/embedding combinations. The one
"correct" auto-mode result (wespeaker on the ES clip, 2/2) is not reliable: the same
embedding/threshold combo got 8 speakers on the EN clip. **Do not trust `num_clusters=-1,
threshold=0.5` as a default.**

Follow-up: I swept the threshold (wespeaker embedding, EN clip, auto mode) to see if this is
fixable without touching pyannote/torch:

| threshold | speakers found |
|---|---|
| 0.5 (default) | 8 |
| 0.65 | 6 |
| 0.70 | 5 |
| 0.80 | 3 |
| 0.90 | 2 ✓ |

It is fixable — raising the threshold to ~0.85–0.90 (wespeaker embedding) recovers the
correct count on the clip that failed worst. This was not re-verified against the ES clip
or against embedding=eres2net (kept the sweep cheap/proportionate), so treat 0.9 as a
starting point to re-tune with real Transcription_24_7 recordings, not a proven-safe final
value.

`num_speakers=2` (i.e. `num_clusters=2` when the count is known, e.g. a 1:1 interview) is
consistently correct, cheap, and should be preferred whenever the caller can supply it.

## Merge quality (word→speaker, max-overlap, grouped into turns)

25-line excerpt, Spanish clip, ASR beam=5 words merged against `eres2net` embedding /
`num_speakers=2` diarization (`Speaker 0` = interviewee, `Speaker 1` = interviewer):

```
Speaker 0 [00:00] me acuerdo que estaba de moda estaba Sipiriti, me acuerdo y me hacían hacer
  cosas así he iniciado y también he llevado pinturas que tenía como cuadros, porque a veces
  hacía cuadros y les ha gustado y me acuerdo de eso de la Copa América, digo perdón del Mundial
Speaker 1 [00:20] cuando trabajo en estos canales ¿con qué personajes famosos tuve por parte?
Speaker 0 [00:28] ¿de Bolivia o de ¿de
Speaker 1 [00:30] Bolivia o del mundo?
Speaker 0 [00:32] no, pues con todas las veces me acuerdo que estaba Alicia, estaba Tonchi,
  estaba este ¿qué se llama? Adolfo Paco, me acuerdo de Miriam Clark de Marcelo Rijel wow de
  esas épocas [...] soy fanático del me he alucinado me he alucinado cuando llegó y otros
  grupos más creo que llegó moro todo bucha y todo eso ¿no?
Speaker 1 [01:22] ¿y de cómo surge eso? ¿de que te gustó el rock?
Speaker 0 [01:26] como
Speaker 1 [01:26] comentabas hace rato ¿te verás de que en el pueblo no había, digamos mucha
  onda mucho de ese tipo de música
Speaker 0 [01:32] lo bueno es que sí o sea no había [...] como Led Zeppelin y yo, Black
  Sabbath esas cosas Pink Floyd [...] ahí es donde he iniciado y me gusta esa música
Speaker 1 [03:01] ¿qué fue lo que hiciste después de trabajar en canales?
Speaker 0 [03:06] no en ahí justamente fue cuando empezaba a pintar hacía aerografía [...]
  hasta el 2005 más o menos en ATV
Speaker 1 [04:14] ¿y esos o sea esos primeros trabajos que has realizado ¿los has dejado
  solamente aquí en el quinto centenario o hay otros lugares donde
Speaker 0 [04:23] no ya no porque ya empezó a mejorar [...] con la tecnología que lo hacen
  en un cacho o sea
Speaker 1 [04:47] con plástico con impresoras tres de ellas
Speaker 0 [04:51] impresoras o no fiesta de alcance ha cambiado harto esas épocas no había
  era difícil tener algo así y chau chau chau Gracias por ver el video.
```

(15 turns total for the full 5-minute clip; `[...]` marks mid-turn elision for this excerpt,
the full untruncated text is in `bench/results/es_results.json` →
`merged_turns_eres2net_num2`.)

## Qualitative judgement

**Spanish ASR accuracy:** good. The model correctly transcribed proper nouns and band names
(Led Zeppelin, Black Sabbath, Pink Floyd, Iron Maiden), Bolivian colloquialisms (chicha,
"homies", "puta" as filler), and kept a long, digression-heavy monologue coherent and
readable. A few clear misses: "Rota Blanca" is almost certainly the Argentine rock band
"Ratabl anca"/"Rata Blanca" mis-heard; "entre el 90 y el 28" is very likely a mis-recognized
year (probably "98"); a couple of filler/interjection words ("wow") look like ASR noise
rather than genuine speech. None of this breaks comprehension of the interview content —
this is normal large-v3-turbo-int8 behavior on informal, regional, fast speech with no
punctuation cues.

**Diarization quality:** workable but fragile. With the speaker count supplied
(`num_speakers=2`), both embedding models correctly split interviewer/interviewee and most
turn boundaries land where a human would put them (at question marks). One real miss: with
the `wespeaker` embedding the short interviewer question at 03:01 ("¿qué fue lo que hiciste
después de trabajar en canales?") got folded into the interviewee's surrounding monologue
instead of being split out (the `eres2net` embedding caught it, shown in the excerpt above).
So even in the "good" configuration, brief interjections during a long monologue can be
missed. **With the speaker count unknown (the realistic case for a general-purpose
recorder), the default auto-threshold is unreliable** (see table above) and must be tuned;
even after tuning, only one embedding/threshold pair was verified fixed, on one clip.

**Bottom line:** this is usable for a first cut of a 2-speaker interview transcript, not a
finished product — turn boundaries need occasional manual correction, and unknown-speaker-
count clustering should not be trusted without further validation on real
Transcription_24_7 recordings.

## Extrapolated wall-clock for longer recordings

> **Superseded for memory, kept for time.** The peak-RSS assumption behind this section — that a
> longer recording costs the same memory as a short one — was wrong, and 40/60-minute jobs were
> OOM-killed. See "Long audio (20/60/120 min)" below for the defect, the windowing fix, and
> *measured* times and peaks at 20, 60 and 120 minutes. The wall-clock estimates here held up
> reasonably well; the memory ones did not.

Using the 5-minute Spanish clip's rates (closer to real meeting-length content than the
2-minute clip) and running ASR beam=5 then the cheapest good diarization config
(`wespeaker`, `num_speakers=2`, RTF 0.088) strictly sequentially (their peak RSS does not
overlap, so this is the actual wall-clock, not a memory-bound estimate):

| Recording length | ASR (beam=5, RTF 0.270) | ASR (beam=1, RTF 0.203) | Diarization (wespeaker, RTF 0.088) | Total (beam=5 + diar) | Total (beam=1 + diar) |
|---|---|---|---|---|---|
| 1 h | 16.2 min | 12.2 min | 5.3 min | 21.5 min | 17.5 min |
| 2 h | 32.4 min | 24.4 min | 10.6 min | 43.0 min | 35.0 min |

(Model load overhead, ~3 s per stage, is negligible at this scale and omitted.)

## Disk and environment

- Python 3.12.14 (pinned via `.python-version`; 3.13 wheels for `ctranslate2`/`onnxruntime`
  used by these packages exist, but 3.12 was chosen up front per the task brief for maximum
  wheel availability — not actually blocked, just the safer choice)
- `.venv/`: 447 MB
- `models/`: 1.6 GB total — `faster-whisper-large-v3-turbo` (int8 CT2): 1.6 GB;
  `sherpa-onnx-diarization/` (segmentation + both embedding models): 71 MB
- `uv 0.12.13`, `faster-whisper 1.2.1`, `ctranslate2 4.8.2`, `sherpa-onnx 1.13.8` +
  `sherpa-onnx-core 1.13.8` (pinned explicitly, see Problems below), `onnxruntime 1.30.0`,
  `numpy 2.5.3`, `soundfile 0.14.0`. No torch, no pyannote, no whisperx anywhere in the
  dependency tree (checked via `uv.lock`).

## Recommended settings (feeds `config.toml` defaults)

- `asr.beam_size = 5` — the extra cost over beam=1 (RTF 0.27 vs 0.20 on the ES clip) is
  small in absolute terms for meeting-length audio and default to quality; expose beam=1 as
  a config override for throughput-constrained days.
- `asr.cpu_threads = 4`, `compute_type = "int8"` — as specified; peak RSS ~1.66 GB regardless
  of beam size, leaves headroom under a 3 GB cap.
- `diarization.embedding_model = wespeaker_en_voxceleb_resnet34.onnx` — faster (~30% lower
  RTF) and much lighter (220–255 MB vs 410–440 MB peak RSS) than eres2net, with equal or
  better speaker-count accuracy once threshold-tuned. (Caveat: eres2net caught one turn
  boundary wespeaker missed on the ES clip — worth re-checking on more data before treating
  this as a strong recommendation rather than a lean one.)
- `diarization.num_clusters = 2` whenever the true speaker count is known (e.g. a 1:1
  client interview) — cheapest and most reliable option, always correct in this benchmark.
- ~~`diarization.cluster_threshold` for the unknown-count fallback: start from 0.85–0.9~~
  **Superseded** by "Auto speaker count" below: no single threshold is safe, because the
  VoxCeleb-English embedding does not separate Spanish speakers at all. Use
  `auto_count = "overcluster"` with `count_embedding_model = eres2net_base_zh.onnx`,
  `overcluster_threshold = 0.15`, `merge_threshold = 0.475`, `min_cluster_share = 0.03`,
  `max_speakers = 8`. `cluster_threshold` now only applies when `auto_count = "threshold"`.
- `diarization.num_threads = 4`, matching ASR, for consistent cgroup CPU accounting.

## Auto speaker count (unknown `num_clusters`) — 2026-09-19

The section above left one thing broken: with the speaker count unknown, sherpa's
`num_clusters=-1` path decides the count from a single cosine threshold and no value of that
threshold is safe. This section is the follow-up investigation, the fix, and its numbers.
Harness: `bench/diar_sweep.py` (cached stages, so the parameter grid is free) and
`bench/diar_e2e.py` (runs the *shipped* `diarize.run()`).

### Dev set

Two clips were not enough to tune anything, so six more were added — 1, 2, 4 and 5 speakers,
Spanish and English, studio and room audio. Sources, licenses and how each ground-truth count
was established (and how confident it is) are in `bench/README.md`. Five of the eight counts
are **certain** from the recording format; three are **estimated** from the source description.

| clip | lang | true k | ground truth |
|---|---|---|---|
| `en_monologo_1spk_4min` | en | 1 | certain (LibriVox solo reading) |
| `es_conferencia_1spk_5min` | es | 1 | certain (solo keynote, room audio) |
| `en_interview_2min` | en | 2 | certain (1:1) |
| `es_entrevista_5min` | es | 2 | certain (1:1) |
| `es_entrevista2_5min` | es | 2 | certain (1:1) |
| `es_mesaredonda_5min` | es | 2 | estimated (TV, presenter + guest) |
| `en_panel4_5min` | en | 4 | estimated (host + 3 named panellists) |
| `es_coloquio_6min` | es | 5 | estimated (moderator + 4 named panellists) |

### What was actually wrong: the embedding, not the threshold

Each clip was over-split by sherpa (threshold 0.15 — far too low on purpose), every resulting
segment was re-embedded with each candidate model, and the segments of each over-cluster were
pooled into one duration-weighted centroid. Then the only question that matters: **how far
apart are two centroids of the same clip?** For a 1-speaker clip every pair is the same voice,
so the max must stay *small*; for a 2+-speaker clip at least one pair is two different voices,
so the max must be *large*. Any gap between those two columns is a usable threshold.

| embedding model | dim | 1-speaker clips: **max** centroid distance | 2+-speaker clips: **min of the max** | best exact count |
|---|---|---|---|---|
| `wespeaker_en_voxceleb_resnet34` (current default) | 256 | 0.209 | 0.224 | 5/8 |
| `wespeaker_en_voxceleb_resnet34_LM` | 256 | 0.348 | 0.240 | 5/8 |
| `3dspeaker_campplus_sv_en_voxceleb_16k` | 512 | 0.537 | 0.522 | 5/8 |
| `3dspeaker_campplus_sv_zh_en_16k-common_advanced` | 192 | 0.213 | 0.713 | 6/8 |
| `3dspeaker_eres2netv2_sv_zh-cn_16k-common` | 192 | 0.184 | 0.409 | 6/8 |
| **`eres2net_base_zh` (3D-Speaker eres2net)** | 512 | **0.348** | **0.760** | **7/8** |

The three VoxCeleb/English-trained models have **no gap at all** — one speaker can look 0.21-0.54
away from himself while two different speakers look 0.22-0.52 apart. No threshold, no linkage and
no clustering algorithm can fix that, which is exactly why the earlier threshold sweep produced
"0.9 works here, 0.8 gives 3 there". The two Mandarin-trained 3D-Speaker models, and eres2net in
particular, open a clean margin on the *same* Spanish and English audio. So the root cause of the
unreliable auto count was the **embedding**, not the clustering.

(`nemo_en_titanet_large` was downloaded and loads, but was lost to the OOM incident in "Problems
hit" #4 and never re-run, so it is missing from this table rather than rejected by it.)

### Method chosen — over-cluster, pool, merge

Implemented in `diarize.py` as `auto_count = "overcluster"`; numpy only, no scipy/scikit-learn:

1. sherpa runs with `overcluster_threshold = 0.15`, i.e. deliberately over-split. It returns
   3-17 over-clusters on the dev set instead of an answer.
2. Every segment ≥ 0.5 s is re-embedded with `count_embedding_model` (eres2net) and the segments
   of each over-cluster are pooled into one duration-weighted, L2-normalised centroid. Only
   over-clusters holding ≥ 2 s of speech seed the decision.
3. The centroids are merged bottom-up (cosine distance, weighted average linkage) and the
   dendrogram is cut at `merge_threshold = 0.475`.
4. Clusters under `min_cluster_share = 3 %` of the speech are folded into their nearest
   surviving cluster; the count is capped at `max_speakers = 8`; the short over-clusters held
   back in step 2 are assigned to the nearest final centroid.
5. Segments are relabelled and consecutive same-speaker segments are glued back together.

Pooling is the second half of the fix: the decision is taken on tens of seconds of speech per
centroid rather than on the ~1 s windows sherpa's own clustering sees.

### Results — predicted speaker count

`bench/diar_e2e.py`, running the shipped `diarize.run()` end to end. "known" is the reference
path (`num_clusters = k`), included to show it is untouched.

| clip | true k | known count | **before** (`auto_count="threshold"`, 0.90) | **after** (`auto_count="overcluster"`) |
|---|---|---|---|---|
| `en_monologo_1spk_4min` | 1 | 1 ✓ | 1 ✓ | **1** ✓ |
| `es_conferencia_1spk_5min` | 1 | 1 ✓ | 1 ✓ | **1** ✓ |
| `en_interview_2min` | 2 | 2 ✓ | 2 ✓ | **2** ✓ |
| `es_entrevista_5min` | 2 | 2 ✓ | **1** ✗ | **2** ✓ |
| `es_entrevista2_5min` | 2 | 2 ✓ | **1** ✗ | **2** ✓ |
| `es_mesaredonda_5min` | 2 | 2 ✓ | **1** ✗ | **2** ✓ |
| `en_panel4_5min` | 4 | 4 ✓ | **2** ✗ | **4** ✓ |
| `es_coloquio_6min` | 5 | 5 ✓ | **1** ✗ | 2 ✗ |
| **exact** | | 8/8 | **3/8** | **7/8** |

The old default did not merely mis-count: at 0.90 it **collapsed every Spanish multi-speaker clip
into a single speaker**, i.e. it silently produced a transcript with no speaker separation at all.
Lowering it does not help — the earlier sweep showed 0.5 shattering the same clips into 8-18
speakers. There is no good value, which is the whole point of the table above.

### Cost

Same machine, same cgroup (`CPUQuota=400%`, `MemoryMax=2G`), box under load, so treat the
absolute RTFs as ~2× the idle-machine figures in the D14 table and compare the columns, not the
rows:

| mode | RTF (min/median/max over the 8 clips) | peak RSS (isolated process, 6-min clip) | speaker-count exact |
|---|---|---|---|
| known count | 0.165 / 0.201 / 0.221 | 248 MB | 8/8 |
| before (`threshold`) | 0.111 / 0.133 / 0.170 | 254 MB | 3/8 |
| **after (`overcluster`)** | **0.203 / 0.234 / 0.259** | **1114 MB** | **7/8** |

**Time.** The new auto path costs **+52 % to +106 % over the old auto path** (median ≈ +75 %) and
about **+15 % over the known-count path**, for +4 correct counts out of 8. The extra work is one
pass of the eres2net extractor over ~20-70 short segments; the pyannote segmentation, which
dominates, is not repeated. The absolute RTFs above were taken with the VPS under load
(`load average` ≈ 7); the same auto run repeated on a quieter box gives **RTF 0.092 / 0.116 /
0.128** (min/median/max, `bench/results/auto_count_e2e_verify.json`, same 7/8 counts) — i.e. about
1.3× the D14 known-count baseline of 0.09, so a 1 h recording goes from ~5.5 min to ~7 min of
diarization, still far below ASR's ~16-26 min. The end-to-end budget in "Extrapolated wall-clock"
is not materially changed.

**Memory is the real cost: 250 MB → 1.11 GB peak.** Holding a second onnxruntime session is most
of it, and eres2net is an expensive second session. Measured on `es_coloquio_6min`, one isolated
process per row:

| `count_embedding_model` | peak RSS | exact |
|---|---|---|
| (same file as `embedding_model`, i.e. wespeaker) | 605 MB | 4/8 — the Spanish clips collapse |
| `3dspeaker_campplus_sv_zh_en_16k-common_advanced` | 477 MB | 6/8 |
| **`eres2net_base_zh` (shipped)** | **1114 MB** | **7/8** |

1.11 GB still sits below ASR's 1.66 GB peak, and the stages run strictly sequentially, so
diarization does not become the binding constraint and `worker.mem_disponible_minima_mb = 3072`
does not need to change. If it ever does, `campplus_sv_zh_en_16k-common_advanced` is the drop-in
lighter option at the cost of one speaker count. The extractor is built inside `diarize()` and
dropped (plus `release_arenas()`) as soon as the embeddings exist, so nothing downstream inherits
the footprint — but the instantaneous peak is unchanged, because sherpa's own session is still
alive while the re-embedding runs.

Two caveats on that 1.11 GB. It is **clip-dependent**: a process that ran the 2-min and 5-min
clips through the same path peaked at 455 MB, while `es_coloquio_6min` alone reaches 1.11 GB
(onnxruntime arena growth, not a leak — the number is a high-water mark). And **nothing here
tests a long recording**: the longest clip is 6 minutes, whereas real jobs run 5 min to 2 h.
sherpa holds the whole waveform and its frame matrices, so memory grows with duration in *both*
the old and the new path; the new path simply starts from a higher base. Measure a 1-2 h
recording before trusting the `MemoryMax` budget in §11.

### How touchy are the defaults?

`merge_threshold` swept over 0.05-1.20 in steps of 0.025 (`diar_sweep.py --stage plateau`),
everything else at the shipped values:

| `merge_threshold` | 0.20-0.40 | **0.425-0.525** | 0.55-0.625 | 0.65 | ≥ 0.675 |
|---|---|---|---|---|---|
| exact | 6/8 | **7/8** | 6/8 | 6/8 | 5/8 |
| what changes | `en_interview` splits into 3-5 | — | `en_panel4` → 3 | `en_panel4` → 2 | speakers start collapsing |

The two 1-speaker clips return **1 across the entire 0.20-0.65 range** — the failure mode that
matters most (chopping a monologue into imaginary speakers) has a very wide safety margin. The
shipped 0.475 sits in the middle of the 7/8 plateau.

`overcluster_threshold` (how hard sherpa over-splits in step 1), best `merge_threshold` for each:

| `overcluster_threshold` | over-clusters produced | best exact | failure |
|---|---|---|---|
| 0.05 | 3-23 | 6/8 | `es_entrevista` splits into 3 — too many one-segment over-clusters |
| **0.15** | 3-17 | **7/8** | — |
| 0.30 | 3-9 | 6/8 | `en_panel4` → 2: step 1 has already merged two panellists, unrecoverable |

The asymmetry matters: over-splitting too little is fatal (step 4 can only merge, never split),
over-splitting too much only adds noise. 0.15 is the middle of that trade, and erring low is the
safe direction.

### Honest limits

- **It degrades on many speakers over a distant mic.** `es_coloquio_6min` (room audio, distant
  mic, a moderator and four panellists in the source) is predicted as **2**. Its label is the
  weakest in the set — forcing `num_clusters` to 1…6 finds only three substantial voices
  (47 % / 32 % / 21 %, unchanged at k=4/5/6), so the excerpt probably holds 3, not 5 — but the
  method is wrong either way, and no setting in the whole grid got it past 3 while keeping the
  rest correct. Treat the auto count as trustworthy for **1-4 speakers on a reasonable mic**, and
  as a *lower bound* on a noisy far-field group recording. `max_speakers = 8` is a cap, not a
  demonstrated capability: nothing here validates 5-8.
- **Short clips.** The smallest clip here is 2 minutes. Below ~1 minute there is simply not
  enough speech per speaker to clear the 2 s pooling floor and the 3 % share, so expect
  under-counting; nothing in this benchmark measures that.
- **A speaker who talks for less than 3 % of the recording is absorbed** by design
  (`min_cluster_share`). In a 1 h meeting that is anyone speaking under ~1.8 min.
- **Three of the eight ground-truth counts are estimated, not verified by listening.** The five
  certain ones are 5/5 correct; the estimated ones are 2/3.
- **Eight clips of 2-6 min is a small dev set**, all from public archives, none of them an actual
  iPhone-in-a-meeting-room recording of the kind this system will receive, and **none longer than
  6 minutes** while real jobs run up to 2 h. The **memory** half of this caveat was the real
  defect and is now fixed and measured at 20/60/120 min — see "Long audio" below. The **accuracy**
  half stands: no speaker count in this document was verified on a recording longer than 6
  minutes, so the counts should still be re-validated on real Transcription 24/7 recordings.
- **The `merge_threshold` default is tuned on this dev set.** The plateau is wide (0.425-0.525)
  and the 1-speaker safety margin is wider still, but 0.475 is not a value derived from theory.
- When the caller *can* supply the count, `num_clusters = k` is still cheaper and better. Nothing
  here changes that path.

## Long audio (20/60/120 min) — 2026-09-19

Everything above was measured on clips of 2-6 minutes, and the "Honest limits" section flagged
that memory on a 1-2 h recording was untested. It was tested, and **it did not work**: this
section is the defect, the fix, and the measured numbers that replace the extrapolations.

### The defect — peak RSS grew with the recording's duration

Both heavy stages handed the *entire* decoded recording to their backend at once:

| stage | what scaled | measured slope | 30 min | 60 min | 120 min |
|---|---|---|---|---|---|
| ASR | Silero VAD state over the whole array, run eagerly inside `transcribe()` before any decoding, on top of the decoded float32 samples | ≈ 10 MB/audio-min (VAD) + 3.8 MB/audio-min (samples) | 2459 MB peak, survived | **OOM-killed after 17 s** | — |
| diarization | `process()` holding the waveform plus its per-frame segmentation output | ≈ 54 MB/audio-min | 1791 MB | ≈ 3.4 GB (extrapolated) | ≈ 6.6 GB (extrapolated) |

Process `VmHWM` at 30 min was already **2588 MB**, above the unit's `MemoryHigh=2500M`. 40 min also
OOMed. The ASR failure is the sharper one because it is not gradual: the VAD pass allocates before
the first segment is emitted, so a 60-min job died 17 s in having produced nothing.

### The fix — one window at a time, one global speaker decision

See ARCHITECTURE.md §3.4/§3.5 "Long audio" (D18) for the full design. In short:

- `windows.plan_windows()` splits the normalized wav into contiguous windows of at most
  `pipeline.window_max_s` (900 s), cutting at the longest low-energy pause found in the last
  `window_cut_search_s` (90 s) of each window, hard at `window_max_s` when the tail is continuous
  speech. The scan is a block-wise RMS read in 30 s chunks — the file is never loaded whole.
- ASR loads the model once and transcribes window by window, offsetting every timestamp; the VAD
  therefore only ever sees one window. Language is detected on the first window with speech and
  pinned. `asr.partial.json` checkpoints after each window.
- Diarization over-clusters each window separately, pools one centroid per `(window, local
  cluster)`, and takes **one global agglomerative decision** over all windows' centroids, so a
  speaker keeps the same label from the first minute to the last. Only centroids cross a window
  boundary; audio and per-segment embeddings do not.

### Regression: short clips are untouched

`bench/diar_e2e.py` re-run on all 8 dev clips after the change (every clip is shorter than
`window_max_s`, so every clip takes the single-window path). Not merely the same *counts* — the
same **segment counts**, clip for clip, against `bench/results/auto_count_e2e.json`:

| clip | true k | before | after | segments before → after |
|---|---|---|---|---|
| `en_monologo_1spk_4min` | 1 | 1 ✓ | 1 ✓ | 8 → 8 |
| `es_conferencia_1spk_5min` | 1 | 1 ✓ | 1 ✓ | 24 → 24 |
| `en_interview_2min` | 2 | 2 ✓ | 2 ✓ | 28 → 28 |
| `es_entrevista_5min` | 2 | 2 ✓ | 2 ✓ | 61 → 61 |
| `es_entrevista2_5min` | 2 | 2 ✓ | 2 ✓ | 12 → 12 |
| `es_mesaredonda_5min` | 2 | 2 ✓ | 2 ✓ | 23 → 23 |
| `en_panel4_5min` | 4 | 4 ✓ | 4 ✓ | 17 → 17 |
| `es_coloquio_6min` | 5 | 2 ✗ | 2 ✗ | 9 → 9 |
| **exact** | | **7/8** | **7/8** | identical |

`known` stayed 8/8 and `auto-old` stayed 3/8, both unchanged — **zero differences across all 24
runs** (8 clips × 3 modes), compared field by field against the stored baseline. This is the point
of pooling the per-window centroids the same way `estimate_speakers()` pools per-segment rows: one
window in, one window out, bit for bit.

Peak RSS over the whole sweep was **1130 MB vs 1134 MB before** — i.e. unchanged. The restructured
single-window path does release sherpa's segmentation session before building the eres2net
extractor (the old code held both), but that does **not** show up as a lower high-water mark,
because the mark is already set during `process()` and `VmHWM` never decreases. Worth stating
plainly: the short-clip peak was *not* improved, only the long-recording behaviour was.

### Measured — real stages, `systemd-run` cgroup, `MemoryMax=3G`, `MemorySwapMax=0`

`bench/long_audio.py`, running the shipped `audio.normalize → asr.run → release → diarize.run →
merge.merge` with `config.toml`'s real settings, no summary call and no Drive delivery, roots
pointed at a scratch directory on disk (**not** tmpfs — a tmpfs artifact is unreclaimable page
cache charged to the same cgroup and would both distort the peak and eat the budget).

Test audio: `bench/audio/es_entrevista2_5min.wav` (2-speaker Spanish interview) concatenated 4 /
12 / 24 times, so the expected speaker count is **2** at every duration and a wrong answer is
obvious. The box was shared and under load throughout, so read the RTFs as an upper bound.

| run | audio | windows | normalize | ASR | diarize | total | RTF | **peak VmHWM** | cgroup `MemoryPeak` | speakers |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) | 20 min | 2 (899 + 301 s) | 0.8 s | 210.6 s (RTF 0.176) | 166.2 s (RTF 0.139) | 377.6 s | 0.315 | **1857 MB** | 1.7 G | **2** ✓ |
| (b) | 60 min | 5 (899/838/900/853/110 s) | 0.8 s | 634.6 s (RTF 0.176) | 578.4 s (RTF 0.161) | 1213.8 s | 0.337 | **1903 MB** | 1.9 G | **2** ✓ |
| (c) | **120 min** | 9 (899/838/900/853/813/881/815/838/363 s) | 4.3 s | 1375.3 s (RTF 0.191) | 1083.3 s (RTF 0.151) | 2462.9 s | 0.342 | **1909 MB** | 2.0 G | **2** ✓ |

**Memory is flat — this is the headline.** Over a **6× range** of recording length the peak moved
**52 MB**:

| audio | 20 min | 60 min | 120 min |
|---|---|---|---|
| peak VmHWM | 1857 MB | 1903 MB | **1909 MB** |
| vs previous row | — | +46 MB | **+6 MB** |
| before this change | 2459 MB at 30 min | **OOM-killed at 17 s** | **never ran** |

The residue is the accumulated `words`/`segments` Python lists (13 962 words at 120 min), not the
models: 60 → 120 min doubles the audio and adds 6 MB. Headroom against `MemoryHigh=2500M` is
**~590 MB at any duration**, and the cgroup peak (2.0 G, page cache included) stays under
`MemoryMax=3G`.

**Time is linear and the per-window cost is constant.** The nine ASR windows of run (c) took
177.1 / 166.3 / 183.9 / 155.8 / 158.1 / 163.3 / 148.6 / 152.7 / 64.7 s (the last is the 363 s
tail) — RTF 0.178-0.204 with **no upward drift** from window 1 to window 9. Window 1 produced
exactly 1746 words in all three runs, window 2 exactly 1642, window 3 exactly 1742: the work done
per window genuinely does not depend on how much audio follows it. End-to-end that is ~20 min of
processing per hour of audio (a 2 h recording took 41 min), on a box that was shared and loaded
throughout.

**Speaker identity survives the joins**, which is the part that windowing could plausibly have
broken:

| run | local clusters (summed over windows) | global speakers | speaker 1 span | speaker 2 span |
|---|---|---|---|---|
| (a) | 25 across 2 windows | **2** | 0.0 – 1198.8 s | 84.8 – 1009.6 s |
| (b) | 63 across 5 windows | **2** | 0.0 – 3598.8 s | 84.8 – 3409.6 s |
| (c) | **127 across 9 windows** | **2** | 0.0 – 7198.9 s | **84.8 – 7009.6 s** |

127 window-local clusters collapsing to exactly 2 global speakers, with **both** of them appearing
on both sides of **all eight** cuts, is the result that matters: the global link is genuinely
re-identifying people across windows rather than inventing a new speaker per window — which is
exactly what per-window clustering plus adjacency stitching would have done (it would have given
up to 127 "speakers", or at best a fresh pair per window). The speech split scales exactly with the
repeated source: 266/24 s for one clip, 1063/96 s at 4×, 3188/289 s at 12×, 6375/579 s at 24×.

A note on the two memory columns: `VmHWM` is the process's anonymous high-water mark and is the
number that matters against `MemoryHigh`; the cgroup's `MemoryPeak` is higher because it also
counts reclaimable page cache for the wav files. Both are reported; neither approaches
`MemoryMax=3G`.

## Problems hit

1. **sherpa-onnx pip wheel doesn't run out of the box.** The `sherpa-onnx==1.13.8` wheel for
   cp312/manylinux dynamically links `libonnxruntime.so` built against onnxruntime's
   internal ABI tag `VERS_1.28.2`, but `pip`/`uv` resolves the `onnxruntime` dependency
   (needed separately by faster-whisper's bundled Silero VAD) to the latest release
   (1.30.0), whose shared library only exports `VERS_1.30.0` — an immediate
   `ImportError: libonnxruntime.so: cannot open shared object file` / then a symbol-version
   mismatch. Fix: explicitly add `sherpa-onnx-core==1.13.8` as a pinned dependency (`uv add
   sherpa-onnx-core==1.13.8`) — that package vendors the exact matching `libonnxruntime.so`
   directly inside the `sherpa_onnx/lib/` directory sherpa-onnx already searches, so no
   `LD_LIBRARY_PATH` hacks or symlinks were needed once it was added properly via uv. This
   is now captured in `pyproject.toml`/`uv.lock`. **`ops/install.sh` for the full system
   (per `docs/ARCHITECTURE.md` §2) should not need an onnxruntime soname fix if it installs
   from this project's `uv.lock`** — flagging in case that doc's note ("fix onnxruntime
   soname") was written against a different workaround.
2. ~~Diarization auto-threshold default is unreliable~~ — **fixed**, see "Auto speaker count"
   above. Root cause was the VoxCeleb-English embedding, not the threshold.
3. No YouTube use (blocked from this VPS, per instructions) — all clips sourced from
   Wikimedia Commons, Internet Archive and LibriVox instead; see `bench/README.md`.
4. Sweeping several embedding models in one process hit the `MemoryMax=2G` cgroup cap
   (`eres2netv2` alone peaks ~1.7 GB, and onnxruntime does not hand the arenas back between
   models). Fixed by running one model per process. Collateral: the `nemo_en_titanet_large`
   run in that same batch exited immediately and was never re-attempted, so titanet is absent
   from the model comparison rather than ruled out — it does download and load correctly.

## Files

- `bench/bench.py` — benchmark harness (ASR + diarization stages, merge, timing/RSS)
- `bench/diar_sweep.py` — auto-speaker-count harness: cached over-split + per-segment
  embeddings, then a free grid search over the clustering parameters
- `bench/diar_e2e.py` — runs the shipped `diarize.run()` over the dev set in all three modes
- `bench/long_audio.py` — long-audio harness: the real `normalize → asr → release → diarize →
  merge` on one file, with per-stage wall time, RTF and `VmHWM` (see "Long audio" above)
- `bench/results/long_audio_{a_20min,b_60min,c_120min}.json` — the three long-audio runs
- `bench/results/auto_count_e2e_windowed.json` — the dev-set regression re-run after windowing
  (compare against `auto_count_e2e.json`: identical on all 24 rows)
- `bench/cache/` — cached segments/embeddings/diagnostics for the sweep (gitignored)
- `bench/results/auto_count_e2e.json` — end-to-end speaker counts, timings and RSS for all three
  modes; `auto_count_e2e_verify.json` — the auto path re-run on a quieter box
- `bench/README.md` — audio provenance and licenses
- `bench/RESULTS.md` — this file
- `bench/audio/*.wav` — test clips (gitignored)
- `bench/results/es_results.json`, `bench/results/en_results.json` — full stage output
  (segments, words, diarization segments, merged turns) for both clips
- `models/faster-whisper-large-v3-turbo/`, `models/sherpa-onnx-diarization/` — model files
  (gitignored). `models/sherpa-onnx-diarization/embedding/` now holds seven candidates
  (~320 MB total) so the comparison table above can be reproduced; only
  `wespeaker_en_resnet34.onnx` (27 MB) and `eres2net_base_zh.onnx` (40 MB) are needed at
  runtime — the other five can be deleted to reclaim ~250 MB.
