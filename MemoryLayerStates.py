class DimensionalMemoryLayer:
    def __init__(self):
        self.layers = {
            "present": [],
            "dream": [],
            "ancestral": [],
            "symbolic": [],
            "encoded": []
        }
        self.timestamps = {layer: [] for layer in self.layers}
        self.max_depth = 100
        self.layer_weights = {layer: 1.0 for layer in self.layers}

    def store(self, value: float, layer: str = "present", timestamp: Optional[str] = None):
        if layer not in self.layers:
            self.layers[layer] = []
            self.timestamps[layer] = []
        
        self.layers[layer].append(value)
        self.timestamps[layer].append(timestamp or datetime.now().isoformat())
        
        if len(self.layers[layer]) > self.max_depth:
            self.layers[layer].pop(0)
            self.timestamps[layer].pop(0)

    def get_layer(self, layer: str) -> List[float]:
        return self.layers.get(layer, [])

    def get_timestamps(self, layer: str) -> List[str]:
        return self.timestamps.get(layer, [])

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

    def set_layer_weight(self, layer: str, weight: float):
        if layer in self.layers:
            self.layer_weights[layer] = max(0.0, min(1.0, weight))

    def weighted_sum(self, layers: List[str]) -> float:
        total = 0.0
        weight_sum = 0.0
        for layer in layers:
            if layer in self.layers and self.layers[layer]:
                weight = self.layer_weights[layer]
                total += weight * self.layers[layer][-1]
                weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def clear_layer(self, layer: str):
        if layer in self.layers:
            self.layers[layer].clear()
            self.timestamps[layer].clear()

    def clear_all(self):
        for layer in self.layers:
            self.clear_layer(layer)

    def get_layer_stats(self, layer: str) -> Dict[str, float]:
        values = self.get_layer(layer)
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    def summarize(self) -> Dict[str, Any]:
        summary = {}
        for k, v in self.layers.items():
            summary[k] = {
                "count": len(v),
                "mean": float(np.mean(v)) if v else 0.0,
                "std": float(np.std(v)) if v else 0.0
            }
        return summary
