class DetectionResult:
    def __init__(self, scores, threshold: float = 0.2):
        self.scores = scores
        self.threshold = float(threshold)
        self.confidence = float(sum(scores) / len(scores)) if scores else 0.0
        self.present = self.confidence > self.threshold

    def __repr__(self):
        return (
            f"DetectionResult("
            f"confidence={self.confidence:.3f}, "
            f"threshold={self.threshold:.3f}, "
            f"present={self.present})"
        )
