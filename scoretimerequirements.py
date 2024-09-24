def sparse_coding_with_time(data, task_complexity='regular'):
    """
    Applies sparse coding to find key features in medical data with time constraints.
    """
    start_time = time.time()

    # Define time limits for different tasks
    if task_complexity == 'regular':
        max_time = 600  # 10 minutes
    elif task_complexity == 'complex':
        max_time = 1800  # 30 minutes
    else:  # Extremely complex
        max_time = 3600  # 1 hour

    coder = SparseCoder(dictionary=data, transform_algorithm='lasso_lars')
    sparse_representation = coder.transform(data)

    elapsed_time = time.time() - start_time
    if elapsed_time > max_time:
        raise TimeoutError(f"Task exceeded maximum allowed time of {max_time} seconds.")
    
    return sparse_representation
