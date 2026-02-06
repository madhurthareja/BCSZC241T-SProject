class WatermarkConfig:
    def __init__(self, key: str, strength: str):
        self.key = key
        self.alpha = self._strength_to_alpha(strength)

    @staticmethod
    def _strength_to_alpha(strength: str) -> float:
        mapping = {
            "low": 0.03,
            "medium": 0.06,
            "high": 0.1,
        }

        if strength not in mapping:
            raise ValueError(f"Invalid strength: {strength}")

        return mapping[strength]
