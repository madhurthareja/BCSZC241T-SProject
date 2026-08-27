from pathlib import Path


class VidmarkFile:
    """Base class for any file-type wrapper in vidmark. Just stores the
    resolved path string so subclasses can build on a common attribute."""

    def __init__(self, path: str | Path):
        # Convert to a plain string once so consumers don't have to deal
        # with Path objects from different callers.
        self.path = str(Path(path))
