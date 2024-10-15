# quantum_problem_solver.py

from problem_analyzer import analyze_problem_description

def quantum_problem_solver(problem_desc, algorithm_registry):
    """
    Solves a quantum problem based on the description.
    """
    # Analyze the problem description
    selected_algorithm = analyze_problem_description(problem_desc, algorithm_registry)

    if selected_algorithm:
        function = selected_algorithm['function']
        args = selected_algorithm.get('args', {})
        description = selected_algorithm.get('description', 'No description available.')

        print(f"Selected Algorithm: {function.__name__}")
        print(f"Description: {description}")
        print("Running the algorithm...")

        result = function(**args)
        if result is not None:
            print(f"Result: {result}")
    else:
        print("No suitable algorithm found for the given problem description.")
        print("Please try rephrasing your description or adding the algorithm to the registry.")
