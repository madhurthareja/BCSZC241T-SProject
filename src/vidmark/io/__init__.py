"""I/O wrappers for video files: readers that expose frames (with PTS-based
sync indexes for sync robustness) and a writer that pipes raw frames into
ffmpeg for a single, controlled H.264 encode."""

from .base import *
from .video_file import *
from .video_writer import *
