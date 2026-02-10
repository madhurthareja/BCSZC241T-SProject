class FrameSelector:
    def should_watermark(self, frame_index: int) -> bool:
        raise NotImplementedError


class AllFrames(FrameSelector):
    def should_watermark(self, frame_index: int) -> bool:
        return True


class EveryNthFrame(FrameSelector):
    def __init__(self, n: int):
        self.n = n

    def should_watermark(self, frame_index: int) -> bool:
        return frame_index % self.n == 0