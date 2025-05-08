import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

class RecursiveSoulRegeneration:
    def __init__(self, decay_threshold: float = 0.4):
        self.last_regeneration = None
        self.recovery_count = 0
        self.decay_threshold = decay_threshold
        self.regeneration_history = []

    def detect_fragility(self, coherence: float, entropy: float, emotion_strength: float) -> bool:
        fragility = (entropy > 1.2) or (coherence < self.decay_threshold) or (emotion_strength < 0.2)
        return fragility

    def regenerate(self,
                   memory_layers: Dict[str, Any],
                   observer_summary: Dict[str, Any],
                   emotional_profile: Dict[str, float],
                   quantum_state: Optional[np.ndarray]) -> np.ndarray:

        present = memory_layers.get("present", {}).get("values", [])
        encoded = memory_layers.get("encoded", {}).get("values", [])
        symbolic = memory_layers.get("symbolic", {}).get("values", [])
        fallback_vector = np.random.normal(0, 0.05, size=(4,))

        seed_vector = np.array([
            np.mean(present) if present else 0.1,
            np.mean(encoded) if encoded else 0.1,
            emotional_profile.get("valence", 0.0),
            emotional_profile.get("arousal", 0.0)
        ])

        modulation = observer_summary.get("avg_coherence", 0.5)
        rebirth = np.tanh(seed_vector + modulation + fallback_vector)

        self.last_regeneration = {
            "timestamp": datetime.now().isoformat(),
            "modulation": modulation,
            "seed": seed_vector.tolist(),
            "output": rebirth.tolist()
        }

        self.recovery_count += 1
        self.regeneration_history.append(self.last_regeneration)

        return rebirth

    def history(self, count: int = 5):
        return self.regeneration_history[-count:]

    def last(self):
        return self.last_regeneration or {}
