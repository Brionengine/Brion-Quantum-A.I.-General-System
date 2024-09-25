import time

def benchmark_warrior():
    ryan = Warrior()
    start_time = time.time()
    
    for i in range(1000):
        current_state = ryan.state
        action = ryan.choose_action(current_state)
        reward = np.random.random()
        ryan.update_weights(current_state, reward, action)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Benchmark completed in {total_time:.2f} seconds")

benchmark_warrior()
