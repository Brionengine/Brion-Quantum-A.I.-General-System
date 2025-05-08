class SoulEvolutionEngine:
    def __init__(self, learning_rate: float = 0.05, decay_rate: float = 0.01):
        self.experiences = []
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.ethical_field = 0.5
        self.patterns = {}

    def absorb_experience(self, coherence: float, entropy: float):
        quality = coherence - entropy
        self.experiences.append(quality)
        if len(self.experiences) > 100:
            self.experiences.pop(0)
        self.ethical_field = self._update_field()

    def _update_field(self):
        weighted = np.array(self.experiences)
        decay = np.exp(-self.decay_rate * np.arange(len(weighted))[::-1])
        field = np.sum(weighted * decay) / np.sum(decay)
        return np.clip(field, 0.0, 1.0)

    def evolve(self, soul_state: Dict[str, Any]):
        vector = np.array(soul_state.get("quantum_state", []))
        if vector.size == 0:
            return None
        modulation = self.ethical_field * self.learning_rate
        evolved = vector + modulation * np.sin(vector)
        return evolved.tolist()

    def snapshot(self):
        return {
            "ethical_field": self.ethical_field,
            "pattern_count": len(self.patterns),
            "experience_depth": len(self.experiences)
        }
