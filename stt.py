import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment, silence
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def format_srt_time(milliseconds):
    """Convert milliseconds into SRT timestamp format (00:00:02,500)."""
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def merge_ranges(ranges):
    """Merge overlapping millisecond ranges."""
    merged = []
    for start_ms, end_ms in sorted(ranges):
        if merged and start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
        else:
            merged.append([start_ms, end_ms])
    return merged


def has_video_stream(input_path):
    """Return whether ffprobe finds a video stream in the input."""
    if shutil.which("ffprobe") is None:
        return False

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def detect_scene_changes(input_path, threshold=0.03, min_interval_ms=800):
    """Detect and debounce visual changes using FFmpeg's scene score."""
    if shutil.which("ffmpeg") is None or not has_video_stream(input_path):
        return []

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("   Warning: FFmpeg scene detection failed; continuing with audio only.")
        return []

    matches = re.findall(
        r"pts_time:([0-9.]+).*?lavfi\.scene_score=([0-9.]+)",
        result.stdout,
        flags=re.DOTALL,
    )
    changes = [(round(float(timestamp) * 1000), float(score)) for timestamp, score in matches]

    # A transition or animation often produces several high-scoring frames.
    # Treat a nearby burst as one change and retain its strongest frame.
    groups = []
    for timestamp_ms, score in changes:
        if groups and timestamp_ms - groups[-1][-1][0] <= min_interval_ms:
            groups[-1].append((timestamp_ms, score))
        else:
            groups.append([(timestamp_ms, score)])

    return [max(group, key=lambda change: change[1])[0] for group in groups]


def find_scene_supported_silences(
    audio,
    scene_changes,
    silence_thresh,
    min_silence_len=150,
    snap_ms=1000,
):
    """Find short audio pauses close enough to corroborate visual changes."""
    if not scene_changes:
        return []

    short_silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    supported = []
    for scene_ms in scene_changes:
        candidates = []
        for start_ms, end_ms in short_silences:
            if start_ms <= scene_ms <= end_ms:
                distance = 0
            else:
                distance = min(abs(scene_ms - start_ms), abs(scene_ms - end_ms))
            if distance <= snap_ms:
                candidates.append((distance, start_ms, end_ms))

        if candidates:
            _, start_ms, end_ms = min(candidates)
            supported.append([start_ms, end_ms])

    return merge_ranges(supported)


def detect_speech_regions(
    audio,
    min_silence_len=700,
    silence_thresh=None,
    additional_silences=None,
    min_scene_speech_ms=500,
):
    """Return absolute millisecond ranges containing speech.

    Whisper's timestamps can attach a final word to a long following silence.
    Detecting the regions from the waveform keeps those pauses independent of
    the transcription model.
    """
    if silence_thresh is None:
        # A relative threshold adapts reasonably well to recordings made at
        # different levels. It can be overridden from the command line.
        silence_thresh = audio.dBFS - 16 if audio.dBFS != float("-inf") else -50

    long_silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10,
    )

    # Scene changes may promote a shorter pause to a cue boundary. Do not use
    # one when it would leave a tiny speech fragment on either side.
    all_silences = merge_ranges(long_silences)
    for start_ms, end_ms in additional_silences or []:
        if any(start_ms <= existing_end and end_ms >= existing_start
               for existing_start, existing_end in all_silences):
            continue
        previous_end = max(
            (existing_end for _, existing_end in all_silences if existing_end < start_ms),
            default=0,
        )
        next_start = min(
            (existing_start for existing_start, _ in all_silences if existing_start > end_ms),
            default=len(audio),
        )
        if (
            start_ms - previous_end >= min_scene_speech_ms
            and next_start - end_ms >= min_scene_speech_ms
        ):
            all_silences = merge_ranges([*all_silences, [start_ms, end_ms]])

    # Speech regions are the complement of all accepted silence ranges.
    regions = []
    cursor_ms = 0
    for start_ms, end_ms in all_silences:
        if start_ms > cursor_ms:
            regions.append([cursor_ms, start_ms])
        cursor_ms = max(cursor_ms, end_ms)
    if cursor_ms < len(audio):
        regions.append([cursor_ms, len(audio)])

    return regions


def audio_segment_to_float_array(audio):
    """Convert a mono AudioSegment to the float waveform Whisper expects."""
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
    max_value = float(1 << (8 * audio.sample_width - 1))
    return samples / max_value


def group_regions_for_transcription(regions, max_gap_ms):
    """Keep regions split by short, scene-assisted pauses in one ASR call."""
    groups = []
    for region in regions:
        if groups and region[0] - groups[-1][-1][1] < max_gap_ms:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def assign_words_to_regions(word_chunks, regions, transcription_start_ms):
    """Assign timestamped Whisper words to absolute waveform regions."""
    region_text = [[] for _ in regions]
    for chunk in word_chunks:
        timestamp = chunk.get("timestamp")
        if not timestamp or timestamp[0] is None:
            continue
        word_start, word_end = timestamp
        word_end = word_start if word_end is None else word_end
        midpoint_ms = transcription_start_ms + round((word_start + word_end) * 500)

        # Normally the midpoint falls inside a speech region. If Whisper puts
        # it inside the pause, choose the closest adjacent region.
        distances = []
        for start_ms, end_ms in regions:
            if start_ms <= midpoint_ms <= end_ms:
                distance = 0
            else:
                distance = min(abs(midpoint_ms - start_ms), abs(midpoint_ms - end_ms))
            distances.append(distance)
        region_index = min(range(len(regions)), key=distances.__getitem__)
        region_text[region_index].append(chunk["text"])

    return ["".join(words).strip() for words in region_text]


def video_to_srt(
    video_path,
    output_srt_path="subtitles.srt",
    model_id="openai/whisper-base.en",
    min_silence_len=700,
    silence_thresh=None,
    speech_padding_ms=100,
    detect_scenes=True,
    scene_threshold=0.03,
    scene_snap_ms=1000,
    scene_min_silence_ms=150,
):
    print("1. Extracting audio from video...")
    audio = AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1)
    effective_silence_thresh = (
        silence_thresh
        if silence_thresh is not None
        else (audio.dBFS - 16 if audio.dBFS != float("-inf") else -50)
    )

    print("2. Detecting audio and visual boundaries...")
    scene_changes = (
        detect_scene_changes(video_path, threshold=scene_threshold)
        if detect_scenes
        else []
    )
    scene_silences = find_scene_supported_silences(
        audio,
        scene_changes,
        silence_thresh=effective_silence_thresh,
        min_silence_len=scene_min_silence_ms,
        snap_ms=scene_snap_ms,
    )
    regions = detect_speech_regions(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=effective_silence_thresh,
        additional_silences=scene_silences,
    )
    if scene_changes:
        print(
            f"   Found {len(scene_changes)} visual change(s) and "
            f"{len(scene_silences)} nearby audio pause(s)."
        )
    print(f"   Found {len(regions)} speech region(s).")

    print(f"3. Loading Hugging Face Whisper model ({model_id})...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Do not use pipeline-level chunk_length_s: Transformers describes it as
    # experimental for seq2seq models. Whisper handles long-form audio itself.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )

    print("4. Transcribing speech regions...")
    cues = []
    transcription_groups = group_regions_for_transcription(regions, min_silence_len)
    region_number = 0
    for group in transcription_groups:
        group_start_ms = group[0][0]
        group_end_ms = group[-1][1]
        # Context padding helps preserve quiet consonants, but the SRT cue uses
        # the unpadded waveform boundaries so none of the gap is consumed.
        transcription_start_ms = max(0, group_start_ms - speech_padding_ms)
        transcription_end_ms = min(len(audio), group_end_ms + speech_padding_ms)
        region_audio = audio[transcription_start_ms:transcription_end_ms]
        samples = audio_segment_to_float_array(region_audio)

        if len(group) == 1:
            # Segment timestamps enable Whisper's native long-form mode when a
            # speech region exceeds 30 seconds. Cue timing still comes from the
            # waveform rather than the model timestamp.
            result = pipe(samples, return_timestamps=True)
            texts = [result["text"].strip()]
        else:
            # Preserve the context across a short visual split, then distribute
            # Whisper's words back to the waveform regions. This avoids asking
            # ASR to understand isolated fragments such as a single final word.
            result = pipe(samples, return_timestamps="word")
            texts = assign_words_to_regions(
                result.get("chunks", []),
                group,
                transcription_start_ms,
            )

        for (start_ms, end_ms), text in zip(group, texts):
            region_number += 1
            if text:
                cues.append({"start": start_ms, "end": end_ms, "text": text})
            else:
                print(f"   Region {region_number} produced no text; skipping it.")

    print(f"5. Generating {output_srt_path}...")
    with open(output_srt_path, "w", encoding="utf-8") as output_file:
        for index, cue in enumerate(cues, start=1):
            output_file.write(f"{index}\n")
            output_file.write(
                f"{format_srt_time(cue['start'])} --> "
                f"{format_srt_time(cue['end'])}\n"
            )
            output_file.write(f"{cue['text']}\n\n")

    print("Done! Your SRT subtitle file is ready with waveform-preserved gaps.")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a video and preserve its silent gaps in an SRT file."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video or audio file")
    parser.add_argument(
        "output_srt_path",
        nargs="?",
        type=Path,
        help="Path to the output SRT file (default: input filename with .srt extension)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=700,
        help="Minimum pause that starts a new subtitle cue (default: 700)",
    )
    parser.add_argument(
        "--silence-thresh-db",
        type=float,
        default=None,
        help="Absolute silence threshold in dBFS (default: recording dBFS minus 16)",
    )
    parser.add_argument(
        "--speech-padding-ms",
        type=int,
        default=100,
        help="Audio context retained around each detected speech region (default: 100)",
    )
    parser.add_argument(
        "--no-scene-detection",
        action="store_true",
        help="Disable visual scene-change hints for video inputs",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.03,
        help="FFmpeg visual-change sensitivity; lower detects subtler changes (default: 0.03)",
    )
    parser.add_argument(
        "--scene-snap-ms",
        type=int,
        default=1000,
        help="Maximum distance from a visual change to a corroborating pause (default: 1000)",
    )
    parser.add_argument(
        "--scene-min-silence-ms",
        type=int,
        default=150,
        help="Shortest pause a visual change may promote to a cue boundary (default: 150)",
    )
    args = parser.parse_args()

    output_srt_path = (
        args.output_srt_path
        if args.output_srt_path is not None
        else Path.cwd() / args.video_path.with_suffix(".srt").name
    )
    video_to_srt(
        str(args.video_path),
        str(output_srt_path),
        min_silence_len=args.min_silence_ms,
        silence_thresh=args.silence_thresh_db,
        speech_padding_ms=args.speech_padding_ms,
        detect_scenes=not args.no_scene_detection,
        scene_threshold=args.scene_threshold,
        scene_snap_ms=args.scene_snap_ms,
        scene_min_silence_ms=args.scene_min_silence_ms,
    )


if __name__ == "__main__":
    main()
