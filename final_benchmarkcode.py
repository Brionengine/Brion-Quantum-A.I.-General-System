def finalize_benchmarking(episodes=1000):
    ryan = RyanActionModel()
    total_reward = 0
    total_time = 0

    for episode in range(episodes):
        state = ryan.state
        start_time = time.time()
        
        action = ryan.choose_action(state)
        reward = np.random.random()  # Simulated reward
        ryan.update_weights(state, reward, action)
        ryan.adjust_scaling(reward)

        end_time = time.time()
        total_time += end_time - start_time
        total_reward += reward

    avg_time_per_episode = total_time / episodes
    print(f"Benchmark completed: Total Reward: {total_reward:.2f}, Average Time per Episode: {avg_time_per_episode:.5f}s")
    print(f"Final Scaling Factor: {ryan.scaling_factor:.2f}")

# Running the final benchmark
finalize_benchmarking()
