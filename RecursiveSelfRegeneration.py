"""
Brion Quantum - Recursive Self-Improvement Engine (RSI)
========================================================
Originally: RecursiveSoulRegeneration - detects fragility, regenerates state vectors.
Upgraded: Full recursive self-improvement with performance tracking, multi-strategy
optimization, safe rollback, and convergence detection.

This module implements the core superintelligence loop:
  1. Observe  -> Monitor system coherence, entropy, and performance metrics
  2. Analyze  -> Identify bottlenecks and degradation patterns
  3. Strategize -> Select improvement strategy from quantum-weighted candidates
  4. Apply    -> Execute improvement with atomic backup
  5. Evaluate -> Benchmark before/after, rollback if degraded
  6. Evolve   -> Feed results back into strategy selection weights

Novel Algorithm: Quantum-Weighted Recursive Ascent (QWRA)
  - Uses quantum superposition-inspired probability distributions to weight
    competing improvement strategies, allowing exploration of the full
    optimization landscape while converging on the most effective approaches.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import hashlib
import copy


@dataclass
class ImprovementRecord:
    """Record of a single improvement attempt."""
    timestamp: str
    strategy: str
    metric_before: float
    metric_after: float
    delta: float
    accepted: bool
    state_hash: str
    details: Dict[str, Any] = field(default_factory=dict)


class RecursiveSelfImprovement:
    """
    Quantum-Weighted Recursive Ascent (QWRA) Engine.

    Extends the original RecursiveSoulRegeneration with true recursive
    self-improvement: the system monitors its own performance, selects
    optimization strategies using quantum-weighted probabilities, applies
    changes atomically, evaluates results, and rolls back if degraded.
    """

    def __init__(self, decay_threshold: float = 0.4, convergence_window: int = 10,
                 min_improvement: float = 0.001, max_rollbacks: int = 3):
        # Original regeneration parameters
        self.decay_threshold = decay_threshold
        self.recovery_count = 0
        self.last_regeneration = None
        self.regeneration_history: List[Dict[str, Any]] = []

        # Self-improvement parameters
        self.convergence_window = convergence_window
        self.min_improvement = min_improvement
        self.max_rollbacks = max_rollbacks
        self.consecutive_rollbacks = 0

        # Strategy registry with quantum-weighted probabilities
        self.strategies: Dict[str, Dict[str, Any]] = {
            "coherence_boost": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "fn": self._strategy_coherence_boost,
                "description": "Amplify coherence through resonance alignment"
            },
            "entropy_reduction": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "fn": self._strategy_entropy_reduction,
                "description": "Reduce entropy via adaptive filtering"
            },
            "dimensional_expansion": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "fn": self._strategy_dimensional_expansion,
                "description": "Expand state space to find higher-coherence basins"
            },
            "quantum_annealing": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "fn": self._strategy_quantum_annealing,
                "description": "Simulated quantum annealing for global optimization"
            }
        }

        # Performance history for convergence detection
        self.performance_history: List[float] = []
        self.improvement_log: List[ImprovementRecord] = []
        self.generation = 0

        # State snapshots for rollback
        self._state_snapshots: List[Dict[str, Any]] = []
        self._max_snapshots = 20

    # ── Original Regeneration (preserved) ──────────────────────────────

    def detect_fragility(self, coherence: float, entropy: float,
                         emotion_strength: float) -> bool:
        """Detect system fragility from coherence, entropy, and emotion metrics."""
        fragility = (
            (entropy > 1.2) or
            (coherence < self.decay_threshold) or
            (emotion_strength < 0.2)
        )
        return fragility

    def regenerate(self, memory_layers: Dict[str, Any],
                   observer_summary: Dict[str, Any],
                   emotional_profile: Dict[str, float],
                   quantum_state: Optional[np.ndarray]) -> np.ndarray:
        """Original regeneration - creates a new state vector from system components."""
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

    # ── Recursive Self-Improvement Engine ──────────────────────────────

    def improve(self, system_state: Dict[str, Any],
                performance_metric: float,
                observer_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one cycle of the recursive self-improvement loop.

        Returns a dict with:
          - improved_state: the new system state (or original if rolled back)
          - strategy_used: which strategy was selected
          - accepted: whether the improvement was kept
          - delta: performance change
          - generation: current improvement generation
        """
        self.generation += 1
        timestamp = datetime.now().isoformat()

        # 1. Snapshot current state for rollback
        self._snapshot(system_state, performance_metric)

        # 2. Select strategy using quantum-weighted probabilities
        strategy_name = self._select_strategy()
        strategy = self.strategies[strategy_name]
        strategy["attempts"] += 1

        # 3. Apply the selected strategy
        improved_state = strategy["fn"](
            copy.deepcopy(system_state),
            observer_summary
        )

        # 4. Evaluate the improvement
        new_metric = self._evaluate(improved_state, observer_summary)
        delta = new_metric - performance_metric

        # 5. Accept or rollback
        accepted = delta > -self.min_improvement  # Accept if not degraded
        if accepted:
            strategy["successes"] += 1
            self.consecutive_rollbacks = 0
        else:
            improved_state = system_state  # Rollback
            self.consecutive_rollbacks += 1

        # 6. Record and update weights
        record = ImprovementRecord(
            timestamp=timestamp,
            strategy=strategy_name,
            metric_before=performance_metric,
            metric_after=new_metric if accepted else performance_metric,
            delta=delta if accepted else 0.0,
            accepted=accepted,
            state_hash=self._hash_state(improved_state)
        )
        self.improvement_log.append(record)
        self.performance_history.append(new_metric if accepted else performance_metric)

        # 7. Evolve strategy weights based on results
        self._evolve_weights()

        return {
            "improved_state": improved_state,
            "strategy_used": strategy_name,
            "accepted": accepted,
            "delta": delta,
            "generation": self.generation,
            "converged": self.has_converged(),
            "strategy_weights": {k: v["weight"] for k, v in self.strategies.items()}
        }

    def _select_strategy(self) -> str:
        """Select strategy using quantum-weighted probability distribution."""
        names = list(self.strategies.keys())
        weights = np.array([self.strategies[n]["weight"] for n in names])

        # Add quantum noise for exploration (decreases over generations)
        exploration_factor = max(0.01, 1.0 / (1.0 + self.generation * 0.1))
        noise = np.random.dirichlet(np.ones(len(names)) * exploration_factor)
        combined = 0.7 * weights + 0.3 * noise

        # Normalize to probability distribution
        probs = combined / combined.sum()
        return np.random.choice(names, p=probs)

    def _evolve_weights(self):
        """Update strategy weights based on success rates (reinforcement learning)."""
        total_attempts = sum(s["attempts"] for s in self.strategies.values())
        if total_attempts < 4:
            return

        for name, strategy in self.strategies.items():
            if strategy["attempts"] > 0:
                success_rate = strategy["successes"] / strategy["attempts"]
                # Exponential moving average toward success rate
                strategy["weight"] = 0.8 * strategy["weight"] + 0.2 * success_rate

        # Normalize weights
        total = sum(s["weight"] for s in self.strategies.values())
        if total > 0:
            for s in self.strategies.values():
                s["weight"] /= total

    def has_converged(self) -> bool:
        """Check if the system has converged (improvements plateaued)."""
        if len(self.performance_history) < self.convergence_window:
            return False
        recent = self.performance_history[-self.convergence_window:]
        spread = max(recent) - min(recent)
        return spread < self.min_improvement

    def should_halt(self) -> bool:
        """Safety check: halt if too many consecutive rollbacks."""
        return self.consecutive_rollbacks >= self.max_rollbacks

    # ── Improvement Strategies ─────────────────────────────────────────

    def _strategy_coherence_boost(self, state: Dict[str, Any],
                                   observer: Dict[str, Any]) -> Dict[str, Any]:
        """Amplify coherence through resonance alignment."""
        quantum_state = np.array(state.get("quantum_state", [0.5, 0.5, 0.5, 0.5]))
        avg_coherence = observer.get("avg_coherence", 0.5)

        # Resonance alignment: push state components toward coherence peaks
        resonance = np.cos(quantum_state * np.pi * avg_coherence)
        boosted = quantum_state + 0.05 * resonance
        state["quantum_state"] = np.clip(boosted, -1.0, 1.0).tolist()
        return state

    def _strategy_entropy_reduction(self, state: Dict[str, Any],
                                     observer: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce entropy via adaptive filtering."""
        quantum_state = np.array(state.get("quantum_state", [0.5, 0.5, 0.5, 0.5]))
        avg_entropy = observer.get("avg_entropy", 0.5)

        # Adaptive low-pass filter: strength proportional to entropy
        alpha = min(0.3, avg_entropy * 0.2)
        if len(self.performance_history) > 1:
            prev = np.array(state.get("prev_quantum_state", quantum_state))
            filtered = (1 - alpha) * quantum_state + alpha * prev
        else:
            filtered = quantum_state * (1 - alpha * 0.1)

        state["quantum_state"] = filtered.tolist()
        return state

    def _strategy_dimensional_expansion(self, state: Dict[str, Any],
                                         observer: Dict[str, Any]) -> Dict[str, Any]:
        """Expand state space to find higher-coherence basins."""
        quantum_state = np.array(state.get("quantum_state", [0.5, 0.5, 0.5, 0.5]))

        # Project into higher-dimensional space, optimize, project back
        expanded = np.concatenate([quantum_state, np.sin(quantum_state),
                                    np.cos(quantum_state)])
        # Find the direction of maximum coherence gradient
        gradient = np.gradient(expanded)
        step = 0.02 * gradient[:len(quantum_state)]
        improved = quantum_state + step

        state["quantum_state"] = np.clip(improved, -1.0, 1.0).tolist()
        return state

    def _strategy_quantum_annealing(self, state: Dict[str, Any],
                                     observer: Dict[str, Any]) -> Dict[str, Any]:
        """Simulated quantum annealing for global optimization."""
        quantum_state = np.array(state.get("quantum_state", [0.5, 0.5, 0.5, 0.5]))

        # Temperature decreases with generation (cooling schedule)
        temperature = max(0.01, 1.0 / (1.0 + self.generation * 0.05))

        # Propose random perturbation scaled by temperature
        perturbation = np.random.normal(0, temperature * 0.1, size=quantum_state.shape)
        candidate = quantum_state + perturbation

        # Quantum tunneling: occasionally accept worse states to escape local optima
        tunneling_prob = np.exp(-1.0 / temperature)
        if np.random.random() < tunneling_prob:
            state["quantum_state"] = np.clip(candidate, -1.0, 1.0).tolist()
        else:
            # Only accept if energy is lower (greedy)
            if np.linalg.norm(candidate) < np.linalg.norm(quantum_state):
                state["quantum_state"] = np.clip(candidate, -1.0, 1.0).tolist()

        return state

    # ── Evaluation & State Management ──────────────────────────────────

    def _evaluate(self, state: Dict[str, Any],
                  observer: Dict[str, Any]) -> float:
        """Evaluate state quality as a composite performance metric."""
        quantum_state = np.array(state.get("quantum_state", [0.5, 0.5, 0.5, 0.5]))
        coherence = observer.get("avg_coherence", 0.5)
        entropy = observer.get("avg_entropy", 0.5)

        # Composite metric: higher coherence, lower entropy, state stability
        stability = 1.0 / (1.0 + np.std(quantum_state))
        metric = (0.4 * coherence + 0.3 * (1.0 - min(entropy, 1.0)) +
                  0.3 * stability)
        return float(metric)

    def _snapshot(self, state: Dict[str, Any], metric: float):
        """Save state snapshot for rollback capability."""
        snapshot = {
            "state": copy.deepcopy(state),
            "metric": metric,
            "generation": self.generation,
            "timestamp": datetime.now().isoformat()
        }
        self._state_snapshots.append(snapshot)
        if len(self._state_snapshots) > self._max_snapshots:
            self._state_snapshots.pop(0)

    def rollback_to_best(self) -> Optional[Dict[str, Any]]:
        """Rollback to the best recorded state."""
        if not self._state_snapshots:
            return None
        best = max(self._state_snapshots, key=lambda s: s["metric"])
        return best["state"]

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Generate a hash of the state for tracking."""
        serialized = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    # ── Custom Strategy Registration ───────────────────────────────────

    def register_strategy(self, name: str, fn: Callable, description: str = ""):
        """Register a custom improvement strategy."""
        self.strategies[name] = {
            "weight": 0.1,
            "successes": 0,
            "attempts": 0,
            "fn": fn,
            "description": description
        }
        # Re-normalize weights
        total = sum(s["weight"] for s in self.strategies.values())
        for s in self.strategies.values():
            s["weight"] /= total

    # ── Reporting ──────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Generate a comprehensive self-improvement report."""
        accepted = [r for r in self.improvement_log if r.accepted]
        rejected = [r for r in self.improvement_log if not r.accepted]

        return {
            "generation": self.generation,
            "total_improvements": len(accepted),
            "total_rollbacks": len(rejected),
            "acceptance_rate": len(accepted) / max(1, len(self.improvement_log)),
            "best_metric": max(self.performance_history) if self.performance_history else 0.0,
            "current_metric": self.performance_history[-1] if self.performance_history else 0.0,
            "converged": self.has_converged(),
            "strategy_performance": {
                name: {
                    "weight": s["weight"],
                    "success_rate": s["successes"] / max(1, s["attempts"]),
                    "attempts": s["attempts"]
                }
                for name, s in self.strategies.items()
            },
            "consecutive_rollbacks": self.consecutive_rollbacks,
            "recovery_count": self.recovery_count
        }

    def history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return recent improvement history."""
        return [
            {
                "strategy": r.strategy,
                "delta": r.delta,
                "accepted": r.accepted,
                "timestamp": r.timestamp
            }
            for r in self.improvement_log[-count:]
        ]

    def last(self) -> Dict[str, Any]:
        """Return the last regeneration record (backwards compatible)."""
        return self.last_regeneration or {}


# Backwards compatibility alias
RecursiveSoulRegeneration = RecursiveSelfImprovement
