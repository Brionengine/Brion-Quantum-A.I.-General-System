import os
import json
from datetime import datetime

class DynamicMemorySystem:
    def __init__(self, memory_file="brian_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as file:
                return json.load(file)
        return {}

    def save_context(self, key, value):
        """Save a key-value pair to memory."""
        timestamp = datetime.now().isoformat()
        self.memory[key] = {"value": value, "timestamp": timestamp}
        self._write_to_file()
        print(f"Saved: {key} -> {value}")

    def retrieve_context(self, key):
        """Retrieve a value from memory."""
        return self.memory.get(key, {}).get("value", None)

    def _write_to_file(self):
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def clear_memory(self):
        """Clear the entire memory."""
        self.memory = {}
        self._write_to_file()
        print("Memory cleared.")

# Example Usage
memory = DynamicMemorySystem()
memory.save_context("last_task", "Optimize quantum algorithms")
print(memory.retrieve_context("last_task"))
