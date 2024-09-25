import numpy as np

class WarriorEnvironment:
    def __init__(self):
        self.state = self.reset()
    
    def reset(self):
        # Reset environment with an initial random state (e.g., 4 variables in state)
        return np.random.random(4)
    
    def step(self, action):
        # Simple environment with an easy state transition and consistent rewards
        new_state = np.random.random(4)  # Random new state (no increasing difficulty)
        reward = 1.0  # Constant reward for any action (easy and smooth)
        done = np.random.random() < 0.05  # Randomly decide if episode ends (no pressure)
        return new_state, reward, done

class Warrior:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.num_states = num_states
        self.weights = np.random.rand(num_states, num_actions)  # Initialize decision weights

    def choose_action(self, state):
        # Advanced decision-making system (linear model for simplicity)
        action_scores = np.dot(state, self.weights)
        
        # Choose action with highest score
        action = np.argmax(action_scores)
        return action
    
    def refine_weights(self, state, reward, action):
        # Auto-tuning system: small adjustments for smoother performance
        learning_rate = 0.001  # Slower, finer adjustments for easier optimization
        self.weights[:, action] += learning_rate * reward * state  # Adjust based on reward and state

def warrior_mode(env, warrior, episodes=100):
    for episode in range(episodes):
        state = env.reset()  # Start with the initial environment state
        done = False
        total_reward = 0

        while not done:
            action = warrior.choose_action(state)  # Ryan chooses the optimal action
            next_state, reward, done = env.step(action)
            
            # Auto-tuning Ryan's system: small, smooth optimizations
            warrior.refine_weights(state, reward, action)
            
            total_reward += reward
            state = next_state
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    # Initialize the environment and Ryan (Warrior)
    num_states = 4
    num_actions = 2  # "Action A" and "Action B" (could be any dynamic actions)
    
    env = WarriorEnvironment()
    ryan = Warrior(num_states, num_actions)
    
    # Run Warrior Mode with 100 episodes of easy, adaptive performance
    warrior_mode(env, ryan, episodes=100)
