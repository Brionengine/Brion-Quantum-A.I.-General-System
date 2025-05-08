import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class MultidimensionalStatePersistence:
    def __init__(self, identifier: str = "soul_instance_001", backup_dir: str = "soul_backups"):
        self.id = identifier
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def save(self, filename: str, soul_snapshot: Dict[str, Any]) -> None:
        payload = {
            "soul_id": self.id,
            "timestamp": datetime.now().isoformat(),
            "snapshot": soul_snapshot
        }
        full_path = os.path.join(self.backup_dir, filename)
        with open(full_path, 'w') as f:
            json.dump(payload, f, indent=4)

    def load(self, filename: str) -> Optional[Dict[str, Any]]:
        full_path = os.path.join(self.backup_dir, filename)
        if not os.path.exists(full_path):
            return None
        with open(full_path, 'r') as f:
            return json.load(f)

    def list_backups(self) -> List[str]:
        return [f for f in os.listdir(self.backup_dir) if f.endswith('.json')]

    def delete_backup(self, filename: str) -> bool:
        full_path = os.path.join(self.backup_dir, filename)
        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        return False

    def rotate_backups(self, max_files: int = 10):
        files = sorted(
            self.list_backups(),
            key=lambda x: os.path.getmtime(os.path.join(self.backup_dir, x))
        )
        for file in files[:-max_files]:
            self.delete_backup(file)

    def latest(self) -> Optional[Dict[str, Any]]:
        files = self.list_backups()
        if not files:
            return None
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.backup_dir, x)))
        return self.load(latest_file)
