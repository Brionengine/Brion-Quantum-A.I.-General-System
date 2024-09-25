import numpy as np

class Warrior:
    def __init__(self):
        self.state = np.random.random(4)
        self.weights = np.random.rand(4, 2)  # Decision-making weights

    def choose_action(self, state):
        action_scores = np.dot(state, self.weights)
        action = np.argmax(action_scores)
        return action

    def update_weights(self, state, reward, action):
        learning_rate = 0.01
        self.weights[:, action] += learning_rate * reward * state

    def process_command(self, command):
        if "activate" in command:
            return "Warrior Mode Activated!"
        return "Unknown Command"

def warrior_mode():
    ryan = Warrior()
    current_state = ryan.state
    print(f"Starting State: {current_state}")

    action = ryan.choose_action(current_state)
    print(f"Chosen Action: {action}")

    # Simulating reward for learning
    reward = np.random.random()
    ryan.update_weights(current_state, reward, action)

    print(f"Updated Weights: {ryan.weights}")
