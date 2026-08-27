"""Frame-selector predicates that decide whether a given sync_index should be
watermarked. Lets callers tune watermark density independently of the
algorithm (e.g. every frame vs. every Nth frame)."""


class FrameSelector:
    """Abstract predicate that decides, given a sync_index, whether the
    watermarker should embed/detect on that frame. Allows watermark density to
    be tuned independently of the algorithm (e.g. watermark every Nth frame)."""

    def should_watermark(self, frame_index: int) -> bool:
        raise NotImplementedError


class AllFrames(FrameSelector):
    """Default selector: watermark every frame."""

    def should_watermark(self, frame_index: int) -> bool:
        return True


class EveryNthFrame(FrameSelector):
    """Watermark one in every N frames (frame_index % n == 0).

    Used to evaluate temporal density vs. robustness trade-offs — fewer
    watermarked frames means lower robustness against frame drops, but
    smaller payload density per second of video."""

    def __init__(self, n: int):
        self.n = n

    def should_watermark(self, frame_index: int) -> bool:
        return frame_index % self.n == 0
