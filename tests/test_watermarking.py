import shutil

import numpy as np
import cv2
import pytest

from vidmark import AllFrames, DctSpreadSpectrumWatermark, Watermarker
from vidmark.core.algorithms import Lfsr, expand_sequence, key_to_bipolar_sequence
from vidmark.io import VideoFile
from vidmark.utils import DetectionResult

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available on PATH"
)


def _make_frames(count: int, size: tuple[int, int] = (64, 64), seed: int = 1234):
    rng = np.random.default_rng(seed)
    height, width = size
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.float32), (height, 1))
    frames = []
    for index in range(count):
        noise = rng.normal(0, 8, size=(height, width)).astype(np.float32)
        base = gradient + noise + (index % 5) * 2.0
        frame = np.stack(
            [base, np.clip(base * 0.8 + 20, 0, 255), np.clip(255 - base, 0, 255)],
            axis=2,
        )
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def _write_video(path, frames, fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_deterministic_watermark_sequence():
    seq_a = expand_sequence(key_to_bipolar_sequence("same-key"), 2)
    seq_b = expand_sequence(key_to_bipolar_sequence("same-key"), 2)
    assert np.array_equal(seq_a, seq_b)


def test_bipolar_sequence_is_actually_bipolar():
    # Regression: bits*2-1 computed in uint8 arithmetic underflows 0 -> 255
    # instead of -1, silently turning half the "bipolar" symbols into +255.
    seq = key_to_bipolar_sequence("any-key")
    assert set(np.unique(seq).tolist()) == {-1, 1}


def test_lfsr_deterministic_and_seed_sensitive():
    items = list(range(18))
    a = Lfsr(seed=0xDEADBEEF).shuffled(items)
    b = Lfsr(seed=0xDEADBEEF).shuffled(items)
    c = Lfsr(seed=0xCAFEF00D).shuffled(items)
    assert a == b
    assert sorted(a) == items
    assert a != c


def test_embedding_transparency_medium_strength():
    frame = _make_frames(1, size=(64, 64), seed=42)[0]
    algorithm = DctSpreadSpectrumWatermark(key="secret", alpha=20.0)
    watermarked = algorithm.apply(frame, 0)
    diff = np.abs(watermarked.astype(np.int16) - frame.astype(np.int16))
    mean_diff = float(diff.mean())
    p99_diff = float(np.percentile(diff, 99))
    assert mean_diff < 6.0
    assert p99_diff < 35.0


def _strong_algorithm(key: str) -> DctSpreadSpectrumWatermark:
    return DctSpreadSpectrumWatermark(
        key=key,
        alpha=40.0,
        sigma_floor=0.0,
        alpha_cap=60.0,
        max_positions=8,
    )


def test_detection_with_correct_key(tmp_path):
    # Test in-memory (no lossy codec round-trip): the mp4v codec quantises away
    # signals smaller than ~8 DCT units, so file I/O is unsuitable for unit testing
    # the algorithm's embed/detect correctness on synthetic low-texture frames.
    frames = _make_frames(24, size=(72, 72), seed=7)
    embed_algo = _strong_algorithm("secret")
    detect_algo = _strong_algorithm("secret")

    scores = []
    for i, frame in enumerate(frames):
        watermarked = embed_algo.apply(frame, i)
        scores.append(detect_algo.detect(watermarked, i))

    result = DetectionResult(scores, threshold=0.01)
    assert result.present
    assert result.confidence > 0.01


@requires_ffmpeg
def test_detection_with_correct_key_real_file_roundtrip(tmp_path):
    # Exercises the actual production path: Watermarker.embed() -> ffmpeg CRF
    # encode -> file read back -> Watermarker.detect(). The in-memory test above
    # proves the algorithm's math; this proves it survives the real codec.
    frames = _make_frames(24, size=(96, 96), seed=5)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "watermarked.mp4"
    _write_video(input_path, frames)

    Watermarker(
        key="correct-key",
        strength="high",
        selector=AllFrames(),
        algorithm=_strong_algorithm("correct-key"),
    ).embed(str(input_path), str(output_path), crf=18)

    result = Watermarker(
        key="correct-key",
        strength="high",
        selector=AllFrames(),
        algorithm=_strong_algorithm("correct-key"),
    ).detect(str(output_path), threshold=0.05)
    assert result.present

    wrong_key_result = Watermarker(
        key="wrong-key",
        strength="high",
        selector=AllFrames(),
        algorithm=_strong_algorithm("wrong-key"),
    ).detect(str(output_path), threshold=0.05)
    assert not wrong_key_result.present


@requires_ffmpeg
def test_detection_with_incorrect_key(tmp_path):
    frames = _make_frames(24, size=(72, 72), seed=11)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "watermarked.mp4"

    _write_video(input_path, frames)
    Watermarker(
        key="correct-key",
        strength="high",
        selector=AllFrames(),
        algorithm=_strong_algorithm("correct-key"),
    ).embed(
        str(input_path), str(output_path)
    )

    result = Watermarker(
        key="wrong-key",
        strength="high",
        selector=AllFrames(),
        algorithm=_strong_algorithm("wrong-key"),
    ).detect(str(output_path), threshold=0.01)
    assert not result.present
    assert abs(result.confidence) < 0.02


def test_multiframe_aggregation_stability(tmp_path):
    # Use _strong_algorithm (sigma_floor=0.0) so all blocks are processed on the
    # synthetic low-texture frames, and run in-memory to avoid codec signal loss.
    frames = _make_frames(30, size=(72, 72), seed=21)
    algo = _strong_algorithm("secret")

    scores = []
    for i, frame in enumerate(frames):
        watermarked = algo.apply(frame, i)
        scores.append(algo.detect(watermarked, i))

    cumulative = np.cumsum(scores) / np.arange(1, len(scores) + 1)
    early_std = float(np.std(cumulative[:10]))
    late_std = float(np.std(cumulative[-10:]))
    assert late_std < early_std


@requires_ffmpeg
def test_frame_drop_requires_reference_fps(tmp_path):
    # Regression for a two-layer bug: (1) a positional frame-index seed
    # desynchronises the moment frames are dropped, and (2) the naive fix
    # (recover index from timestamp) still breaks if it trusts the dropped
    # file's OWN fps metadata, because ffmpeg's variable-frame-rate output
    # rewrites average fps after a drop (e.g. 29.97 -> ~15.15 at 50% drop).
    # Detection must be told the ORIGINAL embedding fps out-of-band.
    import subprocess

    fps = 30.0
    frames = _make_frames(40, size=(96, 96), seed=3)
    input_path = tmp_path / "input.mp4"
    wm_path = tmp_path / "wm.mp4"
    dropped_path = tmp_path / "dropped.mp4"

    _write_video(input_path, frames, fps=fps)
    algo = _strong_algorithm("secret")
    Watermarker(key="secret", strength="high", selector=AllFrames(), algorithm=algo).embed(
        str(input_path), str(wm_path), crf=18
    )

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wm_path),
         "-vf", "select='not(mod(n\\,2))'", "-vsync", "vfr",
         "-c:v", "libx264", "-crf", "18", str(dropped_path)],
        check=True,
    )

    with_correct_fps = Watermarker(
        key="secret", strength="high", selector=AllFrames(), algorithm=algo
    ).detect(str(dropped_path), threshold=0.0, reference_fps=fps)

    without_reference = Watermarker(
        key="secret", strength="high", selector=AllFrames(), algorithm=algo
    ).detect(str(dropped_path), threshold=0.0)

    assert with_correct_fps.confidence > without_reference.confidence


@requires_ffmpeg
def test_sync_search_recovers_dropped_first_frame(tmp_path):
    # Regression for a second desync mode: even with the correct reference_fps,
    # a re-encoder's -vsync vfr output resets its PTS epoch to whichever frame
    # survives first. If frame 0 itself is dropped, every recovered index is
    # off by a constant amount. sync_search tries a window of constant offsets
    # and keeps whichever gives the strongest aggregate correlation.
    import subprocess

    fps = 30.0
    frames = _make_frames(40, size=(96, 96), seed=9)
    input_path = tmp_path / "input.mp4"
    wm_path = tmp_path / "wm.mp4"
    dropped_path = tmp_path / "dropped.mp4"

    _write_video(input_path, frames, fps=fps)
    algo = _strong_algorithm("secret")
    Watermarker(key="secret", strength="high", selector=AllFrames(), algorithm=algo).embed(
        str(input_path), str(wm_path), crf=18
    )

    # Drop frame 0 explicitly (forces the epoch-reset failure mode) plus every
    # 3rd frame after it.
    kept = [i for i in range(1, 40) if i % 3 != 0]
    expr = "+".join(f"eq(n\\,{i})" for i in kept)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wm_path),
         "-vf", f"select='{expr}'", "-vsync", "vfr",
         "-c:v", "libx264", "-crf", "18", str(dropped_path)],
        check=True,
    )

    no_search = Watermarker(
        key="secret", strength="high", selector=AllFrames(), algorithm=algo
    ).detect(str(dropped_path), threshold=0.0, reference_fps=fps, sync_search=0)

    with_search = Watermarker(
        key="secret", strength="high", selector=AllFrames(), algorithm=algo
    ).detect(str(dropped_path), threshold=0.0, reference_fps=fps, sync_search=10)

    assert with_search.confidence > no_search.confidence
