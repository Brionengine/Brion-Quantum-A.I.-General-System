from multiprocessing import Pool

def parallel_map(function, data, processes=None):
    """
    Applies a function to all items in the data iterable in parallel.
    """
    with Pool(processes=processes) as pool:
        result = pool.map(function, data)
    return result
