# get_problem_description.py

def get_problem_description():
    """
    Accepts a natural language input describing the problem.
    Performs basic validation and returns the description.
    """
    while True:
        problem_desc = input("Describe the problem you want the quantum algorithm to solve: ").strip()
        if problem_desc:
            return problem_desc
        else:
            print("Input cannot be empty. Please provide a valid problem description.")
