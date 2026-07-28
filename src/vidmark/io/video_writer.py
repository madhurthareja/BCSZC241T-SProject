import shutil
import subprocess


class VideoWriter:
    """Pipes raw (uncompressed) BGR frames directly into ffmpeg for a single
    H.264 encode at the given CRF.

    Writing frames through OpenCV's mp4v writer and then re-encoding double-
    compresses the signal and quantises the watermark away; piping raw frames
    means the DCT-domain perturbations only ever pass through one lossy
    encode, at a controlled, reproducible CRF.
    """

    def __init__(self, path: str, fps: float, width: int, height: int, crf: int = 18):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; required to write watermarked video")

        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p",
            path,
        ]
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        data = frame.tobytes()
        if len(data) != self._frame_bytes:
            raise ValueError(
                f"frame size {len(data)} bytes does not match expected "
                f"{self._frame_bytes} bytes for {self._width}x{self._height} BGR"
            )
        self._proc.stdin.write(data)

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            stderr = self._proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg exited with code {self._proc.returncode}: {stderr}")
