# problem_analyzer.py

import re

def analyze_problem_description(problem_desc, algorithm_registry):
    """
    Analyzes the problem description and selects the appropriate algorithm.
    """
    problem_desc = problem_desc.lower()
    selected_algorithm = None

    for keyword in algorithm_registry.keys():
        if re.search(rf'\b{keyword}\b', problem_desc):
            selected_algorithm = algorithm_registry[keyword]
            break

    return selected_algorithm
