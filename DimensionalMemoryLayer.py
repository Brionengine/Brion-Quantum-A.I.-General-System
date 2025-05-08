import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class DimensionalMemoryLayer:
    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.layers = {
            "present": [],
            "dream": [],
            "ancestral": [],
            "symbolic": [],
            "encoded": []
        }
        self.timestamps = {
            "present": [],
            "dream": [],
            "ancestral": [],
            "symbolic": [],
            "encoded": []
        }

    def store(self, value: float, layer: str = "present"):
        if layer not in self.layers:
            self.layers[layer] = []
            self.timestamps[layer] = []
        self.layers[layer].append(value)
        self.timestamps[layer].append(datetime.now().isoformat())
        if len(self.layers[layer]) > self.max_depth:
            self.layers[layer].pop(0)
            self.timestamps[layer].pop(0)

    def store_bulk(self, values: List[float], layer: str):
        for v in values:
            self.store(v, layer)

    def get_layer(self, layer: str) -> List[float]:
        return self.layers.get(layer, [])

    def get_recent(self, layer: str, count: int = 10) -> List[float]:
        return self.layers.get(layer, [])[-count:]

    def export_layer(self, layer: str) -> Dict[str, Any]:
        return {
            "values": self.layers.get(layer, [])[-10:],
            "timestamps": self.timestamps.get(layer, [])[-10:]
        }

    def export_all(self) -> Dict[str, Any]:
        export = {}
        for layer in self.layers:
            export[layer] = self.export_layer(layer)
        return export

    def merge_layers(self, target: str, sources: List[str]):
        combined = []
        for src in sources:
            combined.extend(self.layers.get(src, []))
        self.layers[target] = combined[-self.max_depth:]
        self.timestamps[target] = [datetime.now().isoformat()] * len(self.layers[target])

    def prune_layer(self, layer: str, threshold: float):
        values = self.layers.get(layer, [])
        self.layers[layer] = [v for v in values if abs(v) > threshold]
        self.timestamps[layer] = self.timestamps[layer][-len(self.layers[layer]):]

    def normalize_layer(self, layer: str):
        data = np.array(self.layers.get(layer, []))
        if len(data) == 0:
            return
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return
        normalized = (data - mean) / std
        self.layers[layer] = normalized.tolist()

    def layer_similarity(self, a: str, b: str) -> float:
        layer_a = np.array(self.get_layer(a))
        layer_b = np.array(self.get_layer(b))
        if len(layer_a) == 0 or len(layer_b) == 0:
            return 0.0
        min_len = min(len(layer_a), len(layer_b))
        return float(np.corrcoef(layer_a[-min_len:], layer_b[-min_len:])[0, 1])

    def summarize(self) -> Dict[str, Any]:
        summary = {}
        for k, v in self.layers.items():
            summary[k] = {
                "count": len(v),
                "mean": float(np.mean(v)) if v else 0.0,
                "std": float(np.std(v)) if v else 0.0
            }
        return summary
