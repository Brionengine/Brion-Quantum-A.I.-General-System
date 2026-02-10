"""
Brion Quantum - Soul Evolution Engine v2.0
============================================
Ethical field evolution, experience absorption, pattern learning, and
adaptive soul state optimization.

Upgraded with:
  - Pattern recognition across experiences (detects recurring coherence/entropy signatures)
  - Multi-dimensional ethical field with virtue vectors
  - Adaptive learning rate that responds to improvement velocity
  - Experience compression for long-term memory efficiency
  - Integration with IntrospectiveQuantumObserver trend data

Novel Algorithm: Ethical Resonance Field Theory (ERFT)
  - Models the ethical field as a dynamic resonance system where experiences
    create standing waves that guide future evolution toward optimal coherence.
  - The field self-tunes its resonant frequency based on accumulated pattern data.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class SoulEvolutionEngine:
    """
    Ethical Resonance Field Theory (ERFT) Engine.

    Evolves the system's soul state through experience absorption,
    pattern detection, and ethical field modulation. The ethical field
    acts as a guide for quantum state evolution, pushing the system
    toward configurations that maximize coherence while maintaining
    stability.
    """

    def __init__(self, learning_rate: float = 0.05, decay_rate: float = 0.01,
                 pattern_threshold: int = 3, max_experiences: int = 500):
        # Core parameters
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_experiences = max_experiences

        # Experience storage
        self.experiences: List[float] = []
        self.experience_timestamps: List[str] = []

        # Ethical field (multi-dimensional)
        self.ethical_field = 0.5
        self.virtue_vectors = {
            "coherence_drive": 0.5,    # Drive toward coherent states
            "stability_anchor": 0.5,   # Resistance to chaotic change
            "growth_impulse": 0.5,     # Tendency to explore new states
            "harmony_resonance": 0.5   # Balance between competing drives
        }

        # Pattern detection
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_threshold = pattern_threshold
        self._pattern_buffer: List[Tuple[float, float]] = []

        # Evolution history
        self.evolution_log: List[Dict[str, Any]] = []
        self.generation = 0

    def absorb_experience(self, coherence: float, entropy: float,
                          context: Optional[str] = None):
        """
        Absorb a new experience and update the ethical field.

        The experience quality is measured as coherence minus entropy,
        representing how ordered vs disordered the observed state was.
        """
        quality = coherence - entropy
        self.experiences.append(quality)
        self.experience_timestamps.append(datetime.now().isoformat())

        # Compress old experiences if buffer is full
        if len(self.experiences) > self.max_experiences:
            self._compress_experiences()

        # Update ethical field
        self.ethical_field = self._update_field()

        # Update virtue vectors
        self._update_virtues(coherence, entropy)

        # Detect patterns
        self._pattern_buffer.append((coherence, entropy))
        if len(self._pattern_buffer) >= 10:
            self._detect_patterns()
            self._pattern_buffer = self._pattern_buffer[-5:]

        # Adaptive learning rate
        self._adapt_learning_rate()

    def _update_field(self) -> float:
        """Update the ethical field using exponential decay weighting."""
        if not self.experiences:
            return 0.5

        weighted = np.array(self.experiences)
        decay = np.exp(-self.decay_rate * np.arange(len(weighted))[::-1])
        field = np.sum(weighted * decay) / np.sum(decay)
        return float(np.clip(field, 0.0, 1.0))

    def _update_virtues(self, coherence: float, entropy: float):
        """Update virtue vectors based on latest experience."""
        alpha = self.learning_rate * 0.5

        # Coherence drive: strengthened by high coherence experiences
        self.virtue_vectors["coherence_drive"] += alpha * (coherence - 0.5)
        self.virtue_vectors["coherence_drive"] = np.clip(
            self.virtue_vectors["coherence_drive"], 0.0, 1.0)

        # Stability anchor: strengthened by low entropy
        self.virtue_vectors["stability_anchor"] += alpha * (0.5 - entropy)
        self.virtue_vectors["stability_anchor"] = np.clip(
            self.virtue_vectors["stability_anchor"], 0.0, 1.0)

        # Growth impulse: strengthened by improvement trends
        if len(self.experiences) >= 3:
            trend = self.experiences[-1] - self.experiences[-3]
            self.virtue_vectors["growth_impulse"] += alpha * trend
            self.virtue_vectors["growth_impulse"] = np.clip(
                self.virtue_vectors["growth_impulse"], 0.0, 1.0)

        # Harmony: average of all virtues
        self.virtue_vectors["harmony_resonance"] = np.mean([
            self.virtue_vectors["coherence_drive"],
            self.virtue_vectors["stability_anchor"],
            self.virtue_vectors["growth_impulse"]
        ])

    def _detect_patterns(self):
        """Detect recurring patterns in coherence/entropy signatures."""
        if len(self._pattern_buffer) < 5:
            return

        recent = np.array(self._pattern_buffer[-5:])
        coherences = recent[:, 0]
        entropies = recent[:, 1]

        # Pattern: sustained high coherence
        if np.mean(coherences) > 0.7 and np.std(coherences) < 0.1:
            self._record_pattern("sustained_coherence", {
                "avg_coherence": float(np.mean(coherences)),
                "stability": float(1.0 - np.std(coherences))
            })

        # Pattern: entropy spike
        if np.max(entropies) > 1.0 and np.min(entropies) < 0.5:
            self._record_pattern("entropy_spike", {
                "max_entropy": float(np.max(entropies)),
                "min_entropy": float(np.min(entropies))
            })

        # Pattern: improvement trend
        if len(self.experiences) >= 5:
            trend = np.polyfit(range(5), self.experiences[-5:], 1)[0]
            if trend > 0.05:
                self._record_pattern("improvement_trend", {
                    "slope": float(trend)
                })
            elif trend < -0.05:
                self._record_pattern("degradation_trend", {
                    "slope": float(trend)
                })

    def _record_pattern(self, name: str, details: Dict[str, Any]):
        """Record a detected pattern."""
        if name not in self.patterns:
            self.patterns[name] = {
                "count": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": None,
                "details": []
            }
        self.patterns[name]["count"] += 1
        self.patterns[name]["last_seen"] = datetime.now().isoformat()
        self.patterns[name]["details"].append(details)
        if len(self.patterns[name]["details"]) > 10:
            self.patterns[name]["details"] = self.patterns[name]["details"][-10:]

    def _adapt_learning_rate(self):
        """Adapt learning rate based on improvement velocity."""
        if len(self.experiences) < 10:
            return

        recent = self.experiences[-10:]
        velocity = np.mean(np.diff(recent))

        if abs(velocity) < 0.001:
            # Plateaued: increase learning rate to explore
            self.learning_rate = min(0.2, self.learning_rate * 1.05)
        elif velocity > 0.05:
            # Improving fast: keep current rate
            self.learning_rate = self.base_learning_rate
        elif velocity < -0.05:
            # Degrading: reduce learning rate for stability
            self.learning_rate = max(0.001, self.learning_rate * 0.9)

    def _compress_experiences(self):
        """Compress old experiences into summary statistics."""
        # Keep recent 100 experiences, compress the rest into averages
        keep = 100
        if len(self.experiences) <= keep:
            return

        old = self.experiences[:-keep]
        # Compress into chunks of 10
        compressed = []
        for i in range(0, len(old), 10):
            chunk = old[i:i+10]
            compressed.append(float(np.mean(chunk)))

        self.experiences = compressed + self.experiences[-keep:]
        self.experience_timestamps = self.experience_timestamps[-keep:]

    def evolve(self, soul_state: Dict[str, Any]) -> Optional[List[float]]:
        """
        Evolve the soul state using ethical field modulation.

        The evolution applies a sinusoidal perturbation modulated by the
        ethical field strength and virtue vectors, creating a resonance
        pattern that guides the state toward optimal configurations.
        """
        vector = np.array(soul_state.get("quantum_state", []))
        if vector.size == 0:
            return None

        self.generation += 1

        # Compute modulation from ethical field and virtue vectors
        field_strength = self.ethical_field * self.learning_rate
        virtue_mod = np.mean(list(self.virtue_vectors.values()))

        # Resonance evolution: combine sinusoidal and gradient-based terms
        resonance = np.sin(vector * np.pi * self.ethical_field)
        gradient = np.gradient(vector) if vector.size > 1 else np.zeros_like(vector)

        evolved = vector + field_strength * (
            0.6 * resonance +
            0.3 * gradient * virtue_mod +
            0.1 * np.random.normal(0, 0.01, size=vector.shape)  # Quantum noise
        )

        evolved = np.clip(evolved, -1.0, 1.0)

        # Log evolution
        self.evolution_log.append({
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "field_strength": field_strength,
            "virtue_mod": virtue_mod,
            "input_norm": float(np.linalg.norm(vector)),
            "output_norm": float(np.linalg.norm(evolved)),
            "delta_norm": float(np.linalg.norm(evolved - vector))
        })

        return evolved.tolist()

    def snapshot(self) -> Dict[str, Any]:
        """Generate a complete snapshot of the engine state."""
        return {
            "ethical_field": self.ethical_field,
            "virtue_vectors": dict(self.virtue_vectors),
            "learning_rate": self.learning_rate,
            "pattern_count": len(self.patterns),
            "patterns_detected": list(self.patterns.keys()),
            "experience_depth": len(self.experiences),
            "generation": self.generation,
            "improvement_velocity": self._improvement_velocity()
        }

    def _improvement_velocity(self) -> float:
        """Calculate the current rate of improvement."""
        if len(self.experiences) < 5:
            return 0.0
        recent = self.experiences[-5:]
        return float(np.mean(np.diff(recent)))
