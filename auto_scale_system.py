import threading
import time
import psutil  # For system resource monitoring
from concurrent.futures import ThreadPoolExecutor

class AutoScaler:
    def __init__(self, min_threads=2, max_threads=20, cpu_threshold=75, scale_step=2):
        self.min_threads = min_threads
        self.max_threads = max_threads
        self.cpu_threshold = cpu_threshold
        self.scale_step = scale_step
        self.current_threads = min_threads
        self.executor = ThreadPoolExecutor(max_workers=self.current_threads)

    def monitor_and_scale(self):
        """Monitor CPU usage and scale resources dynamically."""
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            print(f"CPU Usage: {cpu_usage}%")

            if cpu_usage > self.cpu_threshold and self.current_threads < self.max_threads:
                self.scale_up()
            elif cpu_usage < self.cpu_threshold * 0.5 and self.current_threads > self.min_threads:
                self.scale_down()

            time.sleep(1)  # Adjust monitoring interval as needed

    def scale_up(self):
        """Scale up the number of threads."""
        self.current_threads = min(self.current_threads + self.scale_step, self.max_threads)
        self.executor._max_workers = self.current_threads  # Dynamically adjust thread count
        print(f"Scaled up to {self.current_threads} threads.")

    def scale_down(self):
        """Scale down the number of threads."""
        self.current_threads = max(self.current_threads - self.scale_step, self.min_threads)
        self.executor._max_workers = self.current_threads  # Dynamically adjust thread count
        print(f"Scaled down to {self.current_threads} threads.")

    def execute_task(self, task, *args):
        """Submit a task to the thread pool."""
        return self.executor.submit(task, *args)

# Example Task
def example_task(task_id):
    print(f"Executing Task {task_id}")
    time.sleep(3)  # Simulate a time-consuming task
    print(f"Task {task_id} completed.")

# Usage Example
if __name__ == "__main__":
    auto_scaler = AutoScaler(min_threads=2, max_threads=10, cpu_threshold=70)

    # Start monitoring and scaling in a separate thread
    threading.Thread(target=auto_scaler.monitor_and_scale, daemon=True).start()

    # Submit tasks dynamically
    for i in range(20):
        auto_scaler.execute_task(example_task, i)
        time.sleep(0.5)  # Simulate user load
