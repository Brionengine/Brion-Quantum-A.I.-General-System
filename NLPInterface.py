# Step 1: NLP Interface to take user input

def get_problem_description():
    # Accept a natural language input describing the problem.
    # This will be later processed to generate quantum circuits
    problem_desc = input("Describe the problem you want the quantum algorithm to solve: ")
    return problem_desc
