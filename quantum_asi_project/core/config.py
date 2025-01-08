import json
import logging
import os

class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        else:
            default_config = {
                "repo_dir": "./tech_repo",
                "log_file": "quantum_asi.log",
                "cycle_time": 60
            }
            self.save_config(default_config)
            return default_config

    def save_config(self, config):
        with open(self.config_file, "w") as file:
            json.dump(config, file, indent=4)
        logging.info(f"Configuration saved: {self.config_file}")

    def get(self, key, default=None):
        return self.config.get(key, default)
