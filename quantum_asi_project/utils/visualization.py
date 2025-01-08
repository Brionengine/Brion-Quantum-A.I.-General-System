from datetime import time
import logging
import matplotlib.pyplot as plt # type: ignore

def visualize_upgrades(upgrade_data):
    """Visualize the number of upgrades per cycle."""
    cycles = list(upgrade_data.keys())
    upgrades = list(upgrade_data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, upgrades, marker='o')
    plt.title("Upgrades per Cycle")
    plt.xlabel("Cycle")
    plt.ylabel("Number of Upgrades")
    plt.grid(True)
    plt.show()

upgrade_data = {}

def infinite_cycle(self):
    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            inefficiencies = self.analyze_code()
            upgraded_files = [self.upgrade_code(file) for file in inefficiencies if file]
            upgraded_files = [file for file in upgraded_files if file]
            upgrade_data[cycle_count] = len(upgraded_files)

            if upgraded_files:
                logging.info(f"Deployed upgrades for files: {', '.join(upgraded_files)}")
            else:
                logging.info("No upgrades needed this cycle.")
            visualize_upgrades(upgrade_data)
            time.sleep(self.cycle_time)
    except KeyboardInterrupt:
        logging.info("Infinite cycle terminated by user.")
