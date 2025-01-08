from concurrent.futures import ThreadPoolExecutor

def process_file_parallel(file_name, asi_instance):
    """Process file using the ASI instance in parallel."""
    return asi_instance.upgrade_code(file_name)

def parallel_processing(files, asi_instance):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda f: process_file_parallel(f, asi_instance), files)
        return list(results)
