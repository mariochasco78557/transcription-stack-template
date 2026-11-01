#!/usr/bin/env python3
"""
Benchmark script for the CPU-only speech pipeline: faster-whisper (ASR) +
sherpa-onnx (offline speaker diarization), merged into speaker-labelled turns.

Design notes:
  - Each stage (ASR beam=5, ASR beam=1, diarization x {clustering mode} x
    {embedding model}) runs in its OWN subprocess (this same script
    re-invoked with --stage <name>). That keeps peak RSS per stage isolated
    (resource.getrusage(RUSAGE_SELF).ru_maxrss inside that subprocess) instead
    of an accumulated whole-process peak across stages. Stages run strictly
    sequentially from the orchestrator, so the process tree's true peak RSS
    at any instant is max(stage_i), not sum(stage_i) -- matching the
    requirement to keep memory bounded under the systemd-run cgroup cap.
  - Orchestrator mode (no --stage) runs all stages, saves a full JSON results
    file, does the word/segment -> speaker merge, and prints a summary.

Usage:
  bench.py --wav path/to.wav --out results_name.json [--threads 4]

Internal (used by the orchestrator itself, not normally called by hand):
  bench.py --wav path/to.wav --stage asr_beam5 --out /tmp/stage.json --threads 4
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FW_MODEL_DIR = MODELS_DIR / "faster-whisper-large-v3-turbo"
SEG_MODEL = MODELS_DIR / "sherpa-onnx-diarization" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODELS = {
    "eres2net": MODELS_DIR / "sherpa-onnx-diarization" / "embedding" / "eres2net_base_zh.onnx",
    "wespeaker": MODELS_DIR / "sherpa-onnx-diarization" / "embedding" / "wespeaker_en_resnet34.onnx",
}

ASR_STAGES = ["asr_beam5", "asr_beam1"]
DIAR_STAGES = [
    "diar_eres2net_auto",
    "diar_eres2net_num2",
    "diar_wespeaker_auto",
    "diar_wespeaker_num2",
]
ALL_STAGES = ASR_STAGES + DIAR_STAGES


def peak_rss_mb() -> float:
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_audio_duration(wav_path: str) -> float:
    import soundfile as sf

    info = sf.info(wav_path)
    return info.frames / info.samplerate


def run_asr_stage(wav_path: str, beam_size: int, threads: int) -> dict:
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        download_root=str(FW_MODEL_DIR),
        cpu_threads=threads,
    )
    t1 = time.perf_counter()
    segments_iter, info = model.transcribe(
        wav_path,
        language=None,  # auto-detect
        beam_size=beam_size,
        vad_filter=True,
        word_timestamps=True,
    )

    segments = []
    words = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                )
    t2 = time.perf_counter()
    load_time = t1 - t0
    process_time = t2 - t1
    wall = t2 - t0
    duration = get_audio_duration(wav_path)
    return {
        "wall_time_s": wall,
        "model_load_time_s": load_time,
        "process_time_s": process_time,
        "audio_duration_s": duration,
        "rtf": process_time / duration if duration else None,
        "rtf_incl_load": wall / duration if duration else None,
        "peak_rss_mb": peak_rss_mb(),
        "language": info.language,
        "language_probability": info.language_probability,
        "beam_size": beam_size,
        "num_segments": len(segments),
        "num_words": len(words),
        "segments": segments,
        "words": words,
    }


def run_diar_stage(wav_path: str, embedding_key: str, mode: str, threads: int) -> dict:
    import soundfile as sf
    import sherpa_onnx

    t0 = time.perf_counter()

    seg_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
        num_threads=threads,
        debug=False,
        provider="cpu",
    )
    emb_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(EMB_MODELS[embedding_key]),
        num_threads=threads,
        debug=False,
        provider="cpu",
    )
    if mode == "auto":
        clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
    elif mode == "num2":
        clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=2)
    else:
        raise ValueError(f"unknown diarization mode {mode!r}")

    diar_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=seg_config,
        embedding=emb_config,
        clustering=clustering_config,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    diarization = sherpa_onnx.OfflineSpeakerDiarization(diar_config)
    t1 = time.perf_counter()

    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    assert sr == diarization.sample_rate, (
        f"wav sample rate {sr} != model expected {diarization.sample_rate}"
    )

    result = diarization.process(samples)
    segs = result.sort_by_start_time()

    t2 = time.perf_counter()
    load_time = t1 - t0
    process_time = t2 - t1
    wall = t2 - t0
    duration = get_audio_duration(wav_path)
    speaker_ids = sorted({s.speaker for s in segs})
    return {
        "wall_time_s": wall,
        "model_load_time_s": load_time,
        "process_time_s": process_time,
        "audio_duration_s": duration,
        "rtf": process_time / duration if duration else None,
        "rtf_incl_load": wall / duration if duration else None,
        "peak_rss_mb": peak_rss_mb(),
        "embedding_model": embedding_key,
        "mode": mode,
        "num_speakers_found": len(speaker_ids),
        "speaker_ids": speaker_ids,
        "segments": [
            {"speaker": s.speaker, "start": s.start, "end": s.end} for s in segs
        ],
    }


def run_stage(stage: str, wav_path: str, threads: int) -> dict:
    if stage == "asr_beam5":
        return run_asr_stage(wav_path, beam_size=5, threads=threads)
    if stage == "asr_beam1":
        return run_asr_stage(wav_path, beam_size=1, threads=threads)
    if stage == "diar_eres2net_auto":
        return run_diar_stage(wav_path, "eres2net", "auto", threads=threads)
    if stage == "diar_eres2net_num2":
        return run_diar_stage(wav_path, "eres2net", "num2", threads=threads)
    if stage == "diar_wespeaker_auto":
        return run_diar_stage(wav_path, "wespeaker", "auto", threads=threads)
    if stage == "diar_wespeaker_num2":
        return run_diar_stage(wav_path, "wespeaker", "num2", threads=threads)
    raise ValueError(f"unknown stage {stage!r}")


def fmt_mmss(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def merge_words_with_diarization(words: list, diar_segments: list) -> list:
    """Assign each ASR word to the diarization speaker with max time overlap.
    Group consecutive same-speaker words into turns.
    Returns a list of {speaker, start, end, text}.
    """

    def overlap(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    turns = []
    current = None
    for w in words:
        best_speaker = None
        best_overlap = -1.0
        for d in diar_segments:
            ov = overlap(w["start"], w["end"], d["start"], d["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = d["speaker"]
        if best_speaker is None:
            # no diarization segments at all; skip assignment
            continue
        if current is not None and current["speaker"] == best_speaker:
            current["end"] = w["end"]
            current["text"] += w["word"]
        else:
            if current is not None:
                turns.append(current)
            current = {
                "speaker": best_speaker,
                "start": w["start"],
                "end": w["end"],
                "text": w["word"],
            }
    if current is not None:
        turns.append(current)
    return turns


def orchestrate(wav_path: str, out_path: str, threads: int, tmp_dir: Path):
    duration = get_audio_duration(wav_path)
    print(f"[bench] audio={wav_path} duration={duration:.1f}s threads={threads}")

    results = {"wav": wav_path, "audio_duration_s": duration, "stages": {}}

    for stage in ALL_STAGES:
        stage_out = tmp_dir / f"{stage}.json"
        print(f"[bench] running stage {stage} in subprocess ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--wav",
                wav_path,
                "--stage",
                stage,
                "--out",
                str(stage_out),
                "--threads",
                str(threads),
            ],
            check=True,
        )
        wall_wrapper = time.perf_counter() - t0
        with open(stage_out) as f:
            stage_result = json.load(f)
        results["stages"][stage] = stage_result
        print(
            f"[bench]   {stage}: load={stage_result['model_load_time_s']:.1f}s "
            f"process={stage_result['process_time_s']:.1f}s "
            f"(subprocess total {wall_wrapper:.1f}s) rtf={stage_result['rtf']:.3f} "
            f"peak_rss={stage_result['peak_rss_mb']:.0f}MB",
            flush=True,
        )

    # Merge: use beam5 ASR words with the diarization config judged most
    # plausible (num2 clustering, eres2net embedding, as a default choice;
    # documented / can be overridden by inspecting the JSON afterwards).
    asr = results["stages"]["asr_beam5"]
    diar = results["stages"]["diar_eres2net_num2"]
    turns = merge_words_with_diarization(asr["words"], diar["segments"])
    results["merged_turns_eres2net_num2"] = turns

    print("\n[bench] === merged transcript (eres2net, num_speakers=2) ===")
    for t in turns:
        print(f"Speaker {t['speaker']} [{fmt_mmss(t['start'])}] {t['text'].strip()}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[bench] full results written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stage", default=None, choices=ALL_STAGES)
    args = ap.parse_args()

    if args.stage:
        result = run_stage(args.stage, args.wav, args.threads)
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False)
        return

    tmp_dir = Path(args.out).resolve().parent / (Path(args.out).stem + "_stages")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    orchestrate(args.wav, args.out, args.threads, tmp_dir)


if __name__ == "__main__":
    main()
