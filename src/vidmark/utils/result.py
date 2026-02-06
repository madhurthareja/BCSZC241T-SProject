class DetectionResult:
    def __init__(self, scores):
        self.scores = scores
        self.confidence = float(sum(scores) / len(scores)) if scores else 0.0
        self.present = self.confidence > 0.2

    def __repr__(self):
        return (
            f"DetectionResult("
            f"confidence={self.confidence:.3f}, "
            f"present={self.present})"
        )
