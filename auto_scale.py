import time
import psutil

def auto_scale_system(min_resources=0.2, max_resources=0.8, task_complexity='regular'):
    """
    Dynamically scales CPU and memory resources and sets max execution time based on task complexity.
    """
    start_time = time.time()

    # Define time limits for different tasks
    if task_complexity == 'regular':
        max_time = 600  # 10 minutes
    elif task_complexity == 'complex':
        max_time = 1800  # 30 minutes
    else:  # Extremely complex
        max_time = 3600  # 1 hour
    
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()

    # If CPU usage is too low, decrease allocated resources
    if cpu_usage < min_resources * 100:
        scale_down_resources()
    # If CPU usage is too high, increase allocated resources
    elif cpu_usage > max_resources * 100:
        scale_up_resources()

    # Same for memory
    if memory_info.percent < min_resources * 100:
        scale_down_memory()
    elif memory_info.percent > max_resources * 100:
        scale_up_memory()

    elapsed_time = time.time() - start_time
    if elapsed_time > max_time:
        raise TimeoutError(f"Task exceeded maximum allowed time of {max_time} seconds.")

    print(f"Task completed in {elapsed_time} seconds, within allowed time.")

def scale_up_resources():
    print("Increasing resources to handle higher load.")
    # Code to request more CPU cores or memory allocation

def scale_down_resources():
    print("Decreasing resources as load is low.")
    # Code to reduce allocated CPU cores or memory usage

def scale_up_memory():
    print("Increasing memory allocation for higher data demands.")
    # Add commands to allocate more memory for Ryan

def scale_down_memory():
    print("Decreasing memory allocation to optimize performance.")
    # Commands to release unused memory
