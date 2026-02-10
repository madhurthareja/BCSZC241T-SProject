from pathlib import Path


class VidmarkFile:
    def __init__(self, path: str | Path):
        self.path = str(Path(path))
