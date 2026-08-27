"""Core watermarking primitives: algorithms, frame selectors, and the
Watermarker façade that ties them together.

Importing from vidmark.core surfaces the public API used by the app/services
layer: WatermarkConfig, Watermarker, DctSpreadSpectrumWatermark,
FrameSelector implementations, and the bipolar-sequence helpers."""

from .algorithms import *
from .selectors import *
from .watermarker import WatermarkConfig, Watermarker