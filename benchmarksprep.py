def run_complex_benchmark():
    ryan = RyanActionModel()
    total_reward = 0
    num_tests = 1000  # Increase the number of actions to simulate complex interaction

    for i in range(num_tests):
        state = ryan.state
        action = ryan.choose_action(state)
        reward = np.random.random()  # Simulating reward for each action
        ryan.update_weights(state, reward, action)
        ryan.adjust_scaling(reward)
        total_reward += reward

    print(f"Benchmark completed. Total Reward: {total_reward:.2f}, Final Scaling Factor: {ryan.scaling_factor:.2f}")

# Run the complex benchmark
run_complex_benchmark()
