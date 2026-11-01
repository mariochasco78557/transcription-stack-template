# Benchmark audio sources

Audio files themselves live in `bench/audio/` and are gitignored (not committed).
This file documents provenance and license for reproducibility.

## Spanish clip — `es_entrevista_5min.wav`

- Source: a public Wikimedia Commons recording (title omitted in this template; not redistributed)
- License: CC BY-SA 4.0 International
- Original duration: 1:04:33 (48kHz stereo Ogg Vorbis, ~45 MB)
- Extraction: 5-minute segment from 00:10:00 to 00:15:00 (chosen to skip any
  intro/outro and land inside the interviewer/interviewee back-and-forth),
  converted to 16 kHz mono PCM WAV with:
  `ffmpeg -ss 00:10:00 -t 00:05:00 -i <orig>.ogg -ar 16000 -ac 1 -c:a pcm_s16le es_entrevista_5min.wav`
- Expected structure: interview format, 2 speakers (interviewer + interviewee).

## English clip — `en_interview_2min.wav`

- Source: a public Internet Archive recording (not redistributed)
- License: CC BY 3.0 New Zealand
- Description: a Creative Commons radio interview (host + guest).
- Original duration: 9:59 (44.1kHz stereo MP3, ~14.4 MB)
- Extraction: 2-minute segment from 00:02:00 to 00:04:00, converted to 16 kHz
  mono PCM WAV with:
  `ffmpeg -ss 00:02:00 -t 00:02:00 -i cc_interview_en.mp3 -ar 16000 -ac 1 -c:a pcm_s16le en_interview_2min.wav`
- Expected structure: interview format, 2 speakers (interviewer + guest).

---

# Speaker-count dev set (added for the unknown-`num_clusters` work)

Six more clips were added so the auto speaker-count estimator could be tuned on
something wider than two 2-speaker interviews. All are 16 kHz mono PCM WAV, all
gitignored, all freely licensed. The ground-truth count `k` below is the count
**inside the extracted excerpt**, which is what `bench/diar_sweep.py` scores against.

## Ground-truth confidence

| confidence | meaning |
|---|---|
| **certain** | the count follows from the recording format itself (a solo reading, a solo lecture, a 1:1 interview) and cannot plausibly be different inside the excerpt |
| **estimated** | taken from the source description (which names the participants) plus the programme format. Treat ±1 (or worse, for `es_coloquio`) as possible. |

`diar_sweep.py --stage gt` (cached in `bench/cache/gt_forced_k.json`) additionally runs each
estimated clip with `num_clusters` forced to 1…6 and prints the resulting speech-time share per
cluster, as a cross-check. **Read it with care**: that diagnostic runs through sherpa's own
clustering with the `wespeaker` embedding — the very embedding this work shows to be poor at
separating speakers (see `RESULTS.md`). It can say "there are at least N distinct voices here"
with some confidence; it cannot say "there are only N".

Conclusions in `RESULTS.md` are weighted towards the **certain** clips.

| file | lang | k | confidence | conditions |
|---|---|---|---|---|
| `en_monologo_1spk_4min.wav` | en | 1 | certain | studio, read speech |
| `es_conferencia_1spk_5min.wav` | es | 1 | certain | lecture hall, room reverb + audience |
| `en_interview_2min.wav` | en | 2 | certain | radio studio |
| `es_entrevista_5min.wav` | es | 2 | certain | field interview |
| `es_entrevista2_5min.wav` | es | 2 | certain | quiet room |
| `es_mesaredonda_5min.wav` | es | 2 | estimated | TV studio |
| `en_panel4_5min.wav` | en | 4 | estimated | radio studio panel |
| `es_coloquio_6min.wav` | es | 5 | estimated | lecture room, distant mic |

## `en_monologo_1spk_4min.wav` — 1 speaker (certain)

- Source: LibriVox / Internet Archive, *The Adventures of Sherlock Holmes*,
  https://archive.org/details/adventures_holmes — file `adventureholmes_01_doyle.mp3`
- License: public domain (LibriVox recordings are PD; the text is PD).
- Extraction: byte range covering ~00:10:00–00:15:00 of the chapter, then
  `ffmpeg -ss 10 -t 240 -i <chunk>.mp3 -ar 16000 -ac 1 -c:a pcm_s16le en_monologo_1spk_4min.wav`
- Ground truth: **1**. A LibriVox chapter file is recorded by a single reader;
  no dialogue, no second voice. Tests "does the estimator over-split one speaker".

## `es_conferencia_1spk_5min.wav` — 1 speaker (certain)

- Source: a public Wikimedia Commons recording (title omitted in this template; not redistributed)
- License: CC BY-SA 4.0 (attribution in the upstream file page).
- Original: 2:15:50, keynote lecture recorded in a hall (audience present, moderator
  only at the start/end).
- Extraction: byte range ≈ minute 35–41, then `ffmpeg -ss 10 -t 300 ... -ar 16000 -ac 1`.
- Ground truth: **1** — the excerpt sits deep inside the keynote itself. This is the
  hard 1-speaker case: reverberant room, distant mic, audience noise.

## `es_entrevista2_5min.wav` — 2 speakers (certain)

- Source: a public Wikimedia Commons recording (title omitted in this template; not redistributed)
- License: CC BY-SA 4.0 (attribution in the upstream file page).
- Original: 14:05, 48 kHz stereo WAV.
- Extraction: `ffmpeg -ss 00:03:00 -t 00:05:00 -i <orig>.wav -ar 16000 -ac 1 -c:a pcm_s16le`
- Ground truth: **2** (interviewer + interviewee).

## `es_mesaredonda_5min.wav` — 2 speakers (estimated)

- Source: a public Wikimedia Commons recording (title omitted in this template; not redistributed)
- License: CC BY 4.0 (television discussion programme).
- Original: 22:16 Ogg Vorbis.
- Extraction: `ffmpeg -ss 00:02:00 -t 00:05:00 ... -ar 16000 -ac 1`
- Ground truth: **2** (presenter + guest). The description names
  only those two; the forced-`k` diagnostic finds two substantial voices (87 % / 13 %) and nothing
  more when `k` is forced up to 6, which is consistent.

## `en_panel4_5min.wav` — 4 speakers (estimated)

- Source: a public Internet Archive radio panel recording (not redistributed)
- License: CC BY 3.0.
- Original: 53:50. Radio studio panel; the description names the host plus three panellists.
- Extraction: byte range ≈ minute 20–26, then `ffmpeg -ss 10 -t 300 ... -ar 16000 -ac 1`.
- Ground truth: **4**. Discussion format with dense turn-taking, so all four are
  expected in a 5-minute excerpt from the middle; "estimated" because that was not
  verified by listening.

## `es_coloquio_6min.wav` — 5 speakers (estimated)

- Source: a public Wikimedia Commons recording (title omitted in this template; not redistributed)
- License: CC BY-SA 4.0 (attribution in the upstream file page).
- Original: 1:53:20. Five distinct speakers (four panellists + moderator) → **5**.
- Extraction: byte range ≈ minute 95–102 (the discussion/Q&A part, after the
  individual presentations), then `ffmpeg -ss 10 -t 360 ... -ar 16000 -ac 1`.
- Ground truth: **5**, estimated — and this is the weakest label in the set. The forced-`k`
  diagnostic finds only **three** substantial voices in the excerpt (47 % / 32 % / 21 %, stable
  when `k` is forced to 4, 5 or 6), so the excerpt most likely contains 3 of the 5 participants.
  It is used as a "many speakers, room audio" stress clip, not as a precision target; the
  method under-counts it either way (it predicts 2).

---

YouTube was not used as a source (blocked from this VPS per task instructions).
Byte-range extraction (`curl -r`) was used for the multi-hundred-MB sources so the
whole file never had to be downloaded; MP3 resynchronises after an arbitrary cut, and
the first ~10 s of each chunk is discarded by the `-ss 10` above.
