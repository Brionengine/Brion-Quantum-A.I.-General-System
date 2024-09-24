def chain_of_thought_processing(problem, medical_data, task_complexity='regular'):
    """
    Implements a chain of thought process to break down complex medical problems with time management.
    """
    start_time = time.time()

    # Define time limits for different tasks
    if task_complexity == 'regular':
        max_time = 600  # 10 minutes
    elif task_complexity == 'complex':
        max_time = 1800  # 30 minutes
    else:  # Extremely complex
        max_time = 3600  # 1 hour

    thoughts = []
    
    # Step 1: Understand the problem
    thoughts.append(f"Understanding the problem: {problem}")
    
    # Step 2: Gather and analyze medical data
    thoughts.append(f"Analyzing patient data: {medical_data}")
    
    # Step 3: Step-by-step reasoning
    thoughts.append(f"Breaking the problem into smaller steps: ")
    for step in range(1, 4):
        thoughts.append(f"Step {step}: Sub-problem analysis...")

        # Check time at each step
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            raise TimeoutError(f"Task exceeded maximum allowed time of {max_time} seconds.")

    # Step 4: Solution
    solution = f"Proposed solution based on analysis and reasoning..."
    thoughts.append(solution)
    
    return thoughts
