import cv2


class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            path, fourcc, fps, (width, height)
        )

    def write(self, frame):
        self._writer.write(frame)

    def close(self):
        self._writer.release()
