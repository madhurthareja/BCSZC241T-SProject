import numpy as np
import cv2

from vidmark import AllFrames, DctSpreadSpectrumWatermark, Watermarker
from vidmark.core.algorithms import expand_sequence, key_to_bipolar_sequence
from vidmark.io import VideoFile


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


def test_embedding_transparency_medium_strength():
    frame = _make_frames(1, size=(64, 64), seed=42)[0]
    algorithm = DctSpreadSpectrumWatermark(key="secret", alpha=0.06)
    watermarked = algorithm.apply(frame, 0)
    diff = np.abs(watermarked.astype(np.int16) - frame.astype(np.int16))
    mean_diff = float(diff.mean())
    p99_diff = float(np.percentile(diff, 99))
    assert mean_diff < 6.0
    assert p99_diff < 35.0


def _strong_algorithm(key: str) -> DctSpreadSpectrumWatermark:
    return DctSpreadSpectrumWatermark(
        key=key,
        alpha=0.12,
        sigma_floor=0.0,
        alpha_cap=0.2,
        max_positions=8,
    )


def test_detection_with_correct_key(tmp_path):
    frames = _make_frames(24, size=(72, 72), seed=7)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "watermarked.mp4"

    _write_video(input_path, frames)
    embed_algo = _strong_algorithm("secret")
    detect_algo = _strong_algorithm("secret")
    wm = Watermarker(
        key="secret",
        strength="high",
        selector=AllFrames(),
        algorithm=embed_algo,
    )
    wm.embed(str(input_path), str(output_path))

    result = Watermarker(
        key="secret",
        strength="high",
        selector=AllFrames(),
        algorithm=detect_algo,
    ).detect(str(output_path), threshold=0.01)
    assert result.present
    assert result.confidence > 0.01


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
    frames = _make_frames(30, size=(72, 72), seed=21)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "watermarked.mp4"

    _write_video(input_path, frames)
    Watermarker(key="secret", strength="high", selector=AllFrames()).embed(
        str(input_path), str(output_path)
    )

    algorithm = DctSpreadSpectrumWatermark(key="secret", alpha=0.1)
    scores = []
    with VideoFile(output_path) as video:
        for frame_index, frame in enumerate(video.frames()):
            scores.append(algorithm.detect(frame, frame_index))

    cumulative = np.cumsum(scores) / np.arange(1, len(scores) + 1)
    early_std = float(np.std(cumulative[:10]))
    late_std = float(np.std(cumulative[-10:]))
    assert late_std < early_std
