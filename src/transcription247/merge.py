"""Word -> speaker assignment, smoothing and turn grouping (ARCHITECTURE.md §3.6, §7).

Stage 4: `diarized -> merged`. Produces `turns.json`
(`dataclasses.asdict(TurnsResult)` is exactly the §3.6 payload).

Pure functions: no I/O, no models, no clock, no randomness — the same inputs always give the same
turns, which is what makes this the one stage covered by a fast unit test (§15 #1).
"""

from __future__ import annotations

import bisect
import logging
import re

from transcription247.config import MergeConfig
from transcription247.models import SCHEMA_VERSION, AsrResult, SpeakerSegment, Turn, TurnsResult, Word

log = logging.getLogger(__name__)

#: Characters that end a sentence, used by the max_turn_s split rule (§7 step 3).
_SENTENCE_END = (".", "?", "!", "…", ":")

#: No space is kept before these when assembling a turn's text (§7 step 4).
_NO_SPACE_BEFORE = re.compile(r"\s+([,.;:!?…\)])")
_MULTISPACE = re.compile(r"\s{2,}")


class MergeError(RuntimeError):
    """The merge cannot produce turns — today only `no_speech` (§13 #7)."""


# --------------------------------------------------------------------------- 1. assignment


def assign_speakers(
    words: list[Word], segments: list[SpeakerSegment], cfg: MergeConfig
) -> list[int]:
    """Assign each word the diarization segment with maximum time overlap (§7 step 1).

    Both lists are sorted, so a moving cursor makes this O(n+m) rather than O(n·m). Ties go to the
    lower segment index. With no positive overlap anywhere: the segment whose centre is nearest the
    word's centre if that distance <= cfg.max_gap_s, else the previous word's speaker, else the
    first segment's speaker, else 0.
    """
    if not words:
        return []
    if not segments:
        return [0] * len(words)

    centres = [(s.start + s.end) / 2.0 for s in segments]
    # Segment centres are increasing for the non-overlapping segments §3.5 promises, but a backend
    # may still emit overlapped speech, so sort explicitly and bisect that.
    order = sorted(range(len(segments)), key=lambda i: (centres[i], i))
    sorted_centres = [centres[i] for i in order]
    speakers: list[int] = []
    cursor = 0  # first segment that can still overlap the current (or a later) word

    for word in words:
        while cursor + 1 < len(segments) and segments[cursor].end <= word.start:
            cursor += 1

        best_speaker = -1
        best_overlap = 0.0
        index = cursor
        while index < len(segments) and segments[index].start < word.end:
            seg = segments[index]
            overlap = min(word.end, seg.end) - max(word.start, seg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker
            index += 1

        if best_speaker < 0:
            best_speaker = _fallback_speaker(
                word, segments, sorted_centres, order, speakers, cfg
            )
        speakers.append(best_speaker)

    return speakers


def _fallback_speaker(
    word: Word,
    segments: list[SpeakerSegment],
    sorted_centres: list[float],
    order: list[int],
    assigned: list[int],
    cfg: MergeConfig,
) -> int:
    """A word that overlaps no segment at all (§7 step 1, second paragraph)."""
    word_centre = (word.start + word.end) / 2.0
    pos = bisect.bisect_left(sorted_centres, word_centre)
    best_index = -1
    best_distance = float("inf")
    for candidate in (pos - 1, pos):
        if not (0 <= candidate < len(sorted_centres)):
            continue
        distance = abs(sorted_centres[candidate] - word_centre)
        segment_index = order[candidate]
        # Strictly closer wins; an exact tie keeps the lower segment index (§7 step 1).
        if distance < best_distance or (distance == best_distance and segment_index < best_index):
            best_distance = distance
            best_index = segment_index
    if best_index >= 0 and best_distance <= cfg.max_gap_s:
        return segments[best_index].speaker
    if assigned:
        return assigned[-1]
    return segments[0].speaker


# --------------------------------------------------------------------------- 2. smoothing


def _runs(labels: list[int]) -> list[tuple[int, int, int]]:
    """Compress labels into (start_index, end_index_exclusive, label) runs."""
    out: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def smooth(word_speakers: list[int], words: list[Word], cfg: MergeConfig) -> list[int]:
    """Absorb very short speaker flips surrounded by one and the same other speaker (§7 step 2).

    A run is absorbed only when it is short in *both* time and words, has a predecessor and a
    successor, and both carry the same label. Repeats until a pass changes nothing or
    cfg.smooth_passes passes have run. Never mutates its inputs.
    """
    labels = list(word_speakers)
    if len(labels) != len(words):
        raise MergeError("smooth(): word_speakers and words have different lengths")

    for _ in range(max(0, cfg.smooth_passes)):
        runs = _runs(labels)
        changed = False
        for position in range(1, len(runs) - 1):
            start, end, label = runs[position]
            previous_label = runs[position - 1][2]
            next_label = runs[position + 1][2]
            if previous_label != next_label or previous_label == label:
                continue
            if (end - start) > cfg.min_run_words:
                continue
            if (words[end - 1].end - words[start].start) >= cfg.min_run_s:
                continue
            for i in range(start, end):
                labels[i] = previous_label
            changed = True
        if not changed:
            break
    return labels


# --------------------------------------------------------------------------- 3./4. turns


def _assemble(words: list[Word]) -> str:
    text = " ".join(w.word.strip() for w in words)
    text = _MULTISPACE.sub(" ", text)
    text = _NO_SPACE_BEFORE.sub(r"\1", text)
    return text.strip()


def _ends_sentence(word: Word) -> bool:
    stripped = word.word.strip()
    return bool(stripped) and stripped.endswith(_SENTENCE_END)


def group_turns(words: list[Word], word_speakers: list[int], cfg: MergeConfig) -> list[Turn]:
    """Group consecutive words into turns (§7 steps 3-4).

    A new turn starts when the label changes, when the silence before the word exceeds
    cfg.turn_gap_s, or when the turn already spans cfg.max_turn_s *and* the previous word ends a
    sentence. `speaker_label` is left empty here and filled by `renumber()`.
    """
    if len(word_speakers) != len(words):
        raise MergeError("group_turns(): word_speakers and words have different lengths")

    turns: list[Turn] = []
    current: list[Word] = []
    current_speaker = -1

    def flush() -> None:
        if not current:
            return
        text = _assemble(current)
        if text:  # §7 step 4: turns that end up empty are dropped
            turns.append(
                Turn(
                    start=current[0].start,
                    end=current[-1].end,
                    speaker=current_speaker,
                    speaker_label="",
                    text=text,
                )
            )

    for index, word in enumerate(words):
        speaker = word_speakers[index]
        if not current:
            current_speaker = speaker
            current = [word]
            continue

        previous = current[-1]
        new_turn = (
            speaker != current_speaker
            or (word.start - previous.end) > cfg.turn_gap_s
            or ((previous.end - current[0].start) >= cfg.max_turn_s and _ends_sentence(previous))
        )
        if new_turn:
            flush()
            current_speaker = speaker
            current = [word]
        else:
            current.append(word)

    flush()
    return turns


# --------------------------------------------------------------------------- 5. renumbering


def renumber(turns: list[Turn]) -> tuple[list[Turn], list[str]]:
    """Label speakers by first appearance in time order (§7 step 5) -> (turns, participantes).

    Raw cluster ids stay in `turn.speaker`; only `speaker_label` is ever rendered.
    """
    mapping: dict[int, str] = {}
    participantes: list[str] = []
    out: list[Turn] = []
    for turn in turns:
        label = mapping.get(turn.speaker)
        if label is None:
            label = f"Speaker {len(participantes) + 1}"
            mapping[turn.speaker] = label
            participantes.append(label)
        out.append(
            Turn(
                start=turn.start,
                end=turn.end,
                speaker=turn.speaker,
                speaker_label=label,
                text=turn.text,
            )
        )
    return out, participantes


# --------------------------------------------------------------------------- entry point


def merge(asr: AsrResult, diar: list[SpeakerSegment], cfg: MergeConfig) -> TurnsResult:
    """Full §7 pipeline: assign -> smooth -> group -> renumber. Deterministic and I/O-free."""
    if not asr.words:
        raise MergeError("no_speech")  # §13 #7
    if not diar:
        # §7 "Degenerate inputs": one speaker; the worker records the matching `warn` event.
        log.warning("merge: no diarization segments; collapsing to a single speaker")

    segments = sorted(diar, key=lambda s: (s.start, s.end))
    words = sorted(asr.words, key=lambda w: (w.start, w.end))

    speakers = assign_speakers(words, segments, cfg)
    speakers = smooth(speakers, words, cfg)
    turns, participantes = renumber(group_turns(words, speakers, cfg))

    return TurnsResult(
        schema=SCHEMA_VERSION,
        job_id=asr.job_id,
        language=asr.language,
        duration_s=asr.duration_s,
        participantes=participantes,
        turns=turns,
    )
