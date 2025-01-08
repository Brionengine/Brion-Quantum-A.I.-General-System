import concurrent.futures
import time

def task(name, duration):
    """Simulate a computational task."""
    print(f"Task {name} started.")
    time.sleep(duration)
    print(f"Task {name} completed.")
    return f"{name} took {duration} seconds"

def parallel_processing(tasks):
    """Execute tasks in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task, name, duration) for name, duration in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

# Example Usage
tasks_to_run = [("Task1", 2), ("Task2", 3), ("Task3", 1)]
results = parallel_processing(tasks_to_run)
print("Results:", results)
