# main.py

from algorithm_registry import algorithm_registry
from quantum_problem_solver import quantum_problem_solver
from get_problem_description import get_problem_description

def main():
    # Get the problem description from the user
    problem_desc = get_problem_description()

    # Solve the quantum problem
    quantum_problem_solver(problem_desc, algorithm_registry)

if __name__ == "__main__":
    main()
