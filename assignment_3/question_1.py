import matplotlib.pyplot as plt
import numpy as np


class MAPFEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.walls = [(5,0), (5,1), (5,2), (4,2),
                      (0,5), (1,5), (2,5), (2,4),
                      (4,9), (4,8), (4,7), (5,7),
                      (7,4), (7,5), (8,5), (9,5)]   

        self.agent_colors = ['blue', 'yellow', 'violet', 'green']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
        self.current_pos = None

    
    def reset(self):
        self.current_pos = [(1,1), (8,1), (8,8), (1,8)]
        return self.current_pos

    
    def step(self, actions):
        rewards = []
        next_positions = []

        for i, (pos, action) in enumerate(zip(self.current_pos, actions)):
            next_pos = self._get_next_position(pos, action)

            if self._is_valid_move(next_pos):
                next_positions.append(next_pos)
            else:
                next_positions.append(pos)

            reward = 0 if next_pos == self.goal_pos[i] else -1
            rewards.append(reward)

        self.current_pos = next_positions
        done = all([pos == goal for pos, goal in zip(next_positions, self.goal_pos)])

        return self.current_pos, rewards, done, {}

    
    def _get_next_position(self, pos, action):
        x, y = pos
        if action == 0:   # Left
            return (x, max(0, y-1))
        elif action == 1: # Right
            return (x, min(self.grid_size-1, y+1))
        elif action == 2: # Up
            return (max(0, x-1), y)
        elif action == 3: # Down
            return (min(self.grid_size-1, x+1), y)
        return (x, y) #stay

    
    def _is_valid_move(self, pos, current_next_positions):
        # Check wall collision
        if pos in self.walls:
            return False
        
        # Check other agent collision
        if pos in current_next_positions:
            return False
            
        return True


    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Draw walls
        for wall in self.walls:
            ax.add_patch(plt.Rectangle(wall, 1, 1, color='grey'))
            
        # Draw agents
        for i, pos in enumerate(self.current_pos):
            ax.add_patch(plt.Rectangle(pos, 1, 1, color=self.agent_colors[i]))
            ax.text(pos[0]+0.5, pos[1]+0.5, str(i), color='black', ha='center', va='center')
            
        # Draw goals
        for i, goal in enumerate(self.goal_pos):
            plt.plot(goal[0]+0.5, goal[1]+0.5, marker='+', color=self.agent_colors[i], 
                    mew=2, ms=10)
            
        plt.show()
        plt.close()



class QLearningAgent:
    def __init__(self, state_size, n_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.q_table = np.zeros((state_size, state_size, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_size = state_size
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 5)
        return np.argmax(self.q_table[state[0], state[1]])
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action
        return np.argmax(self.q_table[state[0], state[1]])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state[0], state[1], action]
        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state[0], state[1], action] = new_value


def main():
    n_episodes = 1000
    max_steps = 100
    grid_size = 10
    n_agents = 4
    
    # Initialize environment and agents
    env = MAPFEnv(grid_size)
    agents = [QLearningAgent(grid_size, 5) for _ in range(n_agents)]
    
    # Training metrics
    episode_rewards = []
    all_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = np.zeros(n_agents)
        
        for step in range(max_steps):
            # Get actions for all agents
            actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
            
            # Take step in environment
            next_state, rewards, done, _ = env.step(actions)
            
            # Update Q-values for each agent
            for i in range(n_agents):
                agents[i].update(state[i], actions[i], rewards[i], next_state[i])
            
            episode_reward += rewards
            state = next_state
            
            if episode % 100 == 0:
                env.render()
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        all_rewards.append(np.mean(episode_reward))
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_reward):.2f}")
    
    return episode_rewards, all_rewards

# Run training
episode_rewards, all_rewards = train()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(all_rewards)
plt.title('Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig('training_progress.png')
plt.show()


if __name__ == '__main__':
    main()


# env = MAPFEnv(10)
# mapf_env.render(agents_pos=[(1,1), (8,1), (8,8), (1,8)])

