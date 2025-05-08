import numpy as np
from typing import Dict, List
from datetime import datetime

class EmotionalFieldProcessor:
    def __init__(self):
        self.last_field = {"valence": 0.0, "arousal": 0.0}
        self.history = []
        self.field_strengths = []

    def encode(self, valence: float, arousal: float) -> float:
        encoded = np.tanh(valence + arousal)
        self.last_field = {
            "valence": valence,
            "arousal": arousal,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(self.last_field)
        self.field_strengths.append(encoded)
        return encoded

    def recent_emotions(self, count: int = 10) -> List[Dict[str, float]]:
        return self.history[-count:]

    def intensity_profile(self, window: int = 10) -> float:
        values = np.array(self.field_strengths[-window:])
        if values.size == 0:
            return 0.0
        return float(np.mean(np.abs(values)))

    def field_vector(self) -> np.ndarray:
        if not self.history:
            return np.zeros(2)
        valence = np.array([h["valence"] for h in self.history[-10:]])
        arousal = np.array([h["arousal"] for h in self.history[-10:]])
        return np.array([np.mean(valence), np.mean(arousal)])

    def peak_state(self) -> Dict[str, float]:
        if not self.field_strengths:
            return {"strength": 0.0}
        peak_idx = int(np.argmax(np.abs(self.field_strengths)))
        return self.history[peak_idx]

    def reset(self):
        self.history.clear()
        self.field_strengths.clear()
