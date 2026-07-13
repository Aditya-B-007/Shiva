from ..models.ScratchEntry import ScratchEntry

class PromotionPolicy:

    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def should_promote(self, entry: ScratchEntry) -> bool:
        return entry.confidence >= self.confidence_threshold
