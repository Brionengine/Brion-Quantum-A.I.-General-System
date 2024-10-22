import os
import subprocess
import sqlite3
import time
import tracemalloc
import psutil  # For system-level benchmarks
import requests  # For web requests during research
from bs4 import BeautifulSoup  # For web scraping

DB_PATH = "skills_tools_database.db"

def initialize_database():
    """Initialize the database to store skills, tools, and benchmarks."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            code TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            skill_id INTEGER,
            cpu_usage REAL,
            memory_usage REAL,
            exec_time REAL,
            status TEXT,
            FOREIGN KEY (skill_id) REFERENCES skills (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT,
            source TEXT,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def store_skill(name, code, status="active"):
    """Store generated skills in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO skills (name, code, status) VALUES (?, ?, ?)
    ''', (name, code, status))
    conn.commit()
    skill_id = cursor.lastrowid
    conn.close()
    return skill_id

def store_research_result(task, source, data):
    """Store research results in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO research_results (task, source, data) VALUES (?, ?, ?)
    ''', (task, source, data))
    conn.commit()
    conn.close()

def benchmark_execution(skill_id, exec_time, cpu_usage, memory_usage, status):
    """Store benchmark results in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO benchmarks (skill_id, exec_time, cpu_usage, memory_usage, status) 
        VALUES (?, ?, ?, ?, ?)
    ''', (skill_id, exec_time, cpu_usage, memory_usage, status))
    conn.commit()
    conn.close()

def research_online(task):
    """Conduct online research by scraping data or accessing APIs."""
    try:
        print(f"Researching task: {task}")
        response = requests.get(f"https://en.wikipedia.org/wiki/{task.replace(' ', '_')}")
        soup = BeautifulSoup(response.content, 'html.parser')
        data = soup.find('p').text  # Extract the first paragraph as research data

        # Store the research result in the database
        store_research_result(task, "Wikipedia", data)
        print(f"Research data collected: {data[:100]}...")  # Show a preview of the research
        return data

    except Exception as e:
        print(f"Research failed: {str(e)}")
        return None

def generate_code(task_description):
    """Generate code dynamically based on task description."""
    if "physics" in task_description:
        return """
import matplotlib.pyplot as plt
import numpy as np

def simulate_gravity():
    t = np.linspace(0, 10, 100)
    y = 0.5 * 9.81 * t**2
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Gravity Simulation')
    plt.show()

simulate_gravity()
"""
    elif "database" in task_description:
        return """
import sqlite3

def create_user_database():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER
        )
    ''')
    conn.commit()
    conn.close()

create_user_database()
"""
    else:
        # Trigger research if the task is unknown
        research_data = research_online(task_description)
        return f"# TODO: Implement {task_description} logic\n# Research Notes: {research_data}"

def execute_code(code):
    """Execute the generated code with benchmarking."""
    tracemalloc.start()
    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        exec(code, globals())
        status = "success"
    except Exception as e:
        status = f"failure: {str(e)}"

    exec_time = time.time() - start_time
    memory_usage = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # In MB
    cpu_usage = process.cpu_percent(interval=1)

    tracemalloc.stop()
    return exec_time, cpu_usage, memory_usage, status

def main():
    initialize_database()

    while True:
        task = input("Describe the task or tool to generate: ")
        code = generate_code(task)
        print("Generated Code:\n", code)

        # Store the generated code in the database
        skill_id = store_skill(task, code)
        print(f"Skill stored with ID: {skill_id}")

        # Execute the generated code and benchmark it
        exec_time, cpu_usage, memory_usage, status = execute_code(code)
        print(f"Execution Status: {status}")
        print(f"Execution Time: {exec_time:.2f}s, CPU Usage: {cpu_usage:.2f}%, Memory Usage: {memory_usage:.2f} MB")

        cont = input("Do you want to generate another tool? (yes/no): ")
        if cont.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
