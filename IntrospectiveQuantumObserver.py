import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class IntrospectiveQuantumObserver:
    def __init__(self):
        self.coherence_log = []
        self.entropy_log = []
        self.alerts = []
        self.analysis_log = []
        self.thresholds = {
            "low_coherence": 0.3,
            "high_entropy": 1.0
        }

    def record(self, coherence: float, entropy: float):
        timestamp = datetime.now().isoformat()
        self.coherence_log.append(coherence)
        self.entropy_log.append(entropy)

        if coherence < self.thresholds["low_coherence"]:
            self.alerts.append({
                "timestamp": timestamp,
                "type": "low_coherence",
                "value": coherence
            })
        if entropy > self.thresholds["high_entropy"]:
            self.alerts.append({
                "timestamp": timestamp,
                "type": "high_entropy",
                "value": entropy
            })

        self.analysis_log.append({
            "timestamp": timestamp,
            "coherence": coherence,
            "entropy": entropy
        })

    def summary(self) -> Dict[str, Any]:
        coherence = np.array(self.coherence_log[-50:])
        entropy = np.array(self.entropy_log[-50:])
        return {
            "avg_coherence": float(np.mean(coherence)) if coherence.size > 0 else 0.0,
            "avg_entropy": float(np.mean(entropy)) if entropy.size > 0 else 0.0,
            "max_entropy": float(np.max(entropy)) if entropy.size > 0 else 0.0,
            "min_coherence": float(np.min(coherence)) if coherence.size > 0 else 0.0,
            "alerts": self.alerts[-10:]
        }

    def report_last(self) -> Dict[str, Any]:
        if not self.analysis_log:
            return {}
        return self.analysis_log[-1]

    def deviation(self) -> float:
        if len(self.coherence_log) < 2:
            return 0.0
        return float(np.std(self.coherence_log))

    def detect_trend(self, window: int = 5) -> str:
        if len(self.coherence_log) < window:
            return "neutral"
        delta = self.coherence_log[-1] - self.coherence_log[-window]
        if delta > 0.1:
            return "rising"
        elif delta < -0.1:
            return "falling"
        else:
            return "stable"

    def reset(self):
        self.coherence_log.clear()
        self.entropy_log.clear()
        self.alerts.clear()
        self.analysis_log.clear()
