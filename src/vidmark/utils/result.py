class DetectionResult:
    """Outcome of running detection over a video: the per-frame correlation
    scores, the threshold used to decide "watermark present", and the rolled-
    up confidence (mean score across frames)."""

    def __init__(self, scores, threshold: float = 0.2):
        # Raw per-frame correlation scores, one entry per watermarked frame.
        self.scores = scores
        # Decision boundary above which we call the watermark "present".
        self.threshold = float(threshold)
        # Mean of the scores, or 0.0 if there were no watermarked frames.
        self.confidence = float(sum(scores) / len(scores)) if scores else 0.0
        # Boolean verdict: confidence crosses the threshold?
        self.present = self.confidence > self.threshold

    def __repr__(self):
        # Compact debug-friendly summary of the three key fields.
        return (
            f"DetectionResult("
            f"confidence={self.confidence:.3f}, "
            f"threshold={self.threshold:.3f}, "
            f"present={self.present})"
        )
