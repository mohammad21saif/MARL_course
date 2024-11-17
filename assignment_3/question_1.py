# import matplotlib.pyplot as plt
# import numpy as np

# class MAPFEnv:
#     def __init__(self, 
#                 grid_size, 
#                 seed,
#                 mode):
#         self.grid_size = grid_size
#         self.walls = [(5,0), (5,1), (5,2), (4,2),
#                       (0,5), (1,5), (2,5), (2,4),
#                       (4,9), (4,8), (4,7), (5,7),
#                       (7,4), (7,5), (8,5), (9,5)]   

#         self.agent_colors = ['blue', 'yellow', 'violet', 'green']
#         self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
#         self.current_pos = None
#         self.mode = mode

#         if seed is not None:
#             np.random.seed(seed)

#     def reset(self):
#         if self.mode == 'random':
#             self.current_pos = [(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)) for _ in range(4)]
#         else:
#             self.current_pos = [(1,1), (8,1), (8,8), (1,8)]
#         return self.current_pos

#     def step(self, actions):
#         rewards = []
#         next_positions = []
    

#         for i, (pos, action) in enumerate(zip(self.current_pos, actions)):
#             next_pos = self._get_next_position(pos, action)

#             if self._is_valid_move(next_pos, next_positions):  
#                 next_positions.append(next_pos)
#             else:
#                 next_positions.append(pos)

#             reward = 0 if next_pos == self.goal_pos[i] else -1
#             rewards.append(reward)

#         self.current_pos = next_positions
#         done = all([pos == goal for pos, goal in zip(next_positions, self.goal_pos)])
#         return self.current_pos, rewards, done, {}

#     def _get_next_position(self, pos, action):
#         x, y = pos
#         if action == 0:   # Left
#             return (x, max(0, y-1))
#         elif action == 1: # Right
#             return (x, min(self.grid_size-1, y+1))
#         elif action == 2: # Up
#             return (max(0, x-1), y)
#         elif action == 3: # Down
#             return (min(self.grid_size-1, x+1), y)
#         return (x, y) #stay

#     def _is_valid_move(self, pos, current_next_positions):
#         if pos in self.walls:
#             return False

#         if pos in current_next_positions:
#             return False

#         return True
    


# class QLearningAgent:
#     def __init__(self, 
#                 state_size, 
#                 n_actions, 
#                 learning_rate=0.03, 
#                 discount_factor=0.99, 
#                 epsilon=0.1):
        
#         self.q_table = np.zeros((state_size, state_size, n_actions))
#         self.policy = np.zeros((state_size, state_size))
#         self.lr = learning_rate
#         self.gamma = discount_factor
#         self.epsilon = epsilon
#         self.state_size = state_size

#     def get_action(self, state):
#         if np.random.random() < self.epsilon:
#             return np.random.randint(0, 5)  # Random action
#         return np.argmax(self.q_table[state[0], state[1]])

#     def update(self, state, action, reward, next_state):
#         old_value = self.q_table[state[0], state[1], action]
#         next_max = np.max(self.q_table[next_state[0], next_state[1]])
#         new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
#         self.q_table[state[0], state[1], action] = new_value
        


# def train(seed):
#     n_episodes = 1000
#     max_steps = 500
#     grid_size = 10
#     n_agents = 4
    
#     mode = input("Enter mode (random or None): ")

#     env = MAPFEnv(grid_size, seed, mode)
#     agents = [QLearningAgent(grid_size, 5) for _ in range(n_agents)]
    
#     # Initialize agent rewards tracking
#     agent_rewards = np.zeros((n_episodes, n_agents))  # Store rewards for each agent in each episode
    
#     for episode in range(n_episodes):
#         state = env.reset()
#         episode_reward = np.zeros(n_agents)
        
#         for step in range(max_steps):
#             actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
#             next_state, rewards, done, _ = env.step(actions)
            
#             for i in range(n_agents):
#                 agents[i].update(state[i], actions[i], rewards[i], next_state[i])
            
#             episode_reward += rewards
#             state = next_state
            
#             if done:
#                 break
        
#         agent_rewards[episode] = episode_reward  # Track rewards for all agents
        
#         if episode % 100 == 0:
#             print(f"Episode {episode}, Average Reward: {np.mean(episode_reward):.2f}")
    
#     return agent_rewards, agents, env

# def find_min_steps_for_all_agents(env, agents):
#     """
#     Function to find the minimum number of steps for all agents to reach their goal positions
#     using the QLearning agents after training.
#     """
#     steps_to_goal = np.zeros(len(agents))  # Store steps for each agent
    
#     for i, agent in enumerate(agents):
#         state = env.reset()  # Reset environment and get initial positions of all agents
#         agent_position = state[i]  # Get the starting position of this agent
#         steps = 0
        
#         while agent_position != env.goal_pos[i]:  # Loop until the agent reaches its goal
#             action = agent.get_action(agent_position)
#             next_state, _, _, _ = env.step([action if j == i else 0 for j in range(len(agents))])  # Get next state
#             agent_position = next_state[i]  # Update agent's position
#             steps += 1
        
#         steps_to_goal[i] = steps  # Store the number of steps for this agent
    
#     return steps_to_goal

# # Run training
# SEED = 42

# # Run training with seed
# agent_rewards, agents, env = train(seed=SEED)

# # Plot reward for each agent
# plt.figure(figsize=(12, 6))
# for i in range(agent_rewards.shape[1]):
#     plt.plot(agent_rewards[:, i], label=f'Agent {i}')

# plt.title('Reward per Episode for Each Agent')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend()
# plt.savefig('agent_rewards.png')
# plt.show()

# # Find minimum steps for all agents to reach their goal positions
# steps = find_min_steps_for_all_agents(env, agents)
# print(f"Minimum steps for each agent to reach their goal: {steps}")

import matplotlib.pyplot as plt
import numpy as np

class MAPFEnv:
    # [Previous MAPFEnv implementation remains the same]
    def __init__(self, 
                grid_size, 
                seed,
                mode):
        self.grid_size = grid_size
        self.walls = [(5,0), (5,1), (5,2), (4,2),
                      (0,5), (1,5), (2,5), (2,4),
                      (4,9), (4,8), (4,7), (5,7),
                      (7,4), (7,5), (8,5), (9,5)]   

        self.agent_colors = ['blue', 'yellow', 'violet', 'green']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
        self.current_pos = None
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        if self.mode == 'random':
            self.current_pos = [(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)) for _ in range(4)]
        else:
            self.current_pos = [(1,1), (8,1), (8,8), (1,8)]
        return self.current_pos

    def step(self, actions):
        rewards = []
        next_positions = []
    
        for i, (pos, action) in enumerate(zip(self.current_pos, actions)):
            next_pos = self._get_next_position(pos, action)

            if self._is_valid_move(next_pos, next_positions):  
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
        if pos in self.walls:
            return False

        if pos in current_next_positions:
            return False

        return True

class QLearningAgent:
    def __init__(self, 
                state_size, 
                n_actions, 
                learning_rate=0.03, 
                discount_factor=0.99, 
                epsilon=0.1):
        
        self.q_table = np.zeros((state_size, state_size, n_actions))
        self.policy = np.zeros((state_size, state_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_size = state_size

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action
        return self.policy[state[0], state[1]]

    # def update(self, state, action, reward, next_state, current_positions, agent_idx, env):
    #     """
    #     Updated Q-value update method incorporating improvements from the second implementation
    #     """
    #     # Get current Q-value
    #     curr_x, curr_y = int(state[0]), int(state[1])
    #     next_x, next_y = int(next_state[0]), int(next_state[1])
        
    #     # Get current Q-value
    #     current_q = self.q_table[curr_x, curr_y, action]
        
    #     # Get maximum Q-value for next state
    #     next_max_q = np.max(self.q_table[next_x, next_y])
        
    #     # Calculate temporal difference
    #     td_error = reward + self.gamma * next_max_q - current_q
        
    #     # Update Q-value
    #     new_q = current_q + self.lr * td_error
        
    #     # Update Q-table
    #     self.q_table[curr_x, curr_y, action] = new_q
        
    #     # Update policy
    #     self.policy[curr_x, curr_y] = np.argmax(self.q_table[curr_x, curr_y])

    def update(self, state, action, reward, next_state, current_positions, agent_idx, env):
        """
        Updated Q-value update method with robust type checking
        
        Parameters:
        state: position coordinates
        action: integer representing the action taken
        reward: float representing the reward received
        next_state: position coordinates of next state
        """
        action = int(action)
        curr_x, curr_y = map(int, state)
        next_x, next_y = map(int, next_state)
        
        current_q = self.q_table[curr_x, curr_y, action]
        next_max_q = np.max(self.q_table[next_x, next_y])
        td_error = reward + self.gamma * next_max_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[curr_x, curr_y, action] = new_q
        self.policy[curr_x, curr_y] = np.argmax(self.q_table[curr_x, curr_y])


def train(seed):
    n_episodes = 8000
    max_steps = 4000
    grid_size = 10
    n_agents = 4
    
    mode = input("Enter mode (random or None): ")

    env = MAPFEnv(grid_size, seed, mode)
    agents = [QLearningAgent(grid_size, 5) for _ in range(n_agents)]
    
    # Initialize agent rewards tracking
    agent_rewards = np.zeros((n_episodes, n_agents))
    cumulative_rewards = np.zeros((n_episodes, n_agents))
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = np.zeros(n_agents)
        
        for step in range(max_steps):
            actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
            next_state, rewards, done, _ = env.step(actions)
            
            # Update each agent's Q-values
            for i in range(n_agents):
                agents[i].update(
                    state[i], 
                    actions[i], 
                    rewards[i], 
                    next_state[i],
                    state,  # Current positions of all agents
                    i,      # Agent index
                    env     # Environment instance for collision checking
                )
            
            episode_reward += rewards
            state = next_state
            
            if done:
                break
        
        # Store episode rewards
        agent_rewards[episode] = episode_reward
        
        # Update cumulative rewards
        if episode == 0:
            cumulative_rewards[episode] = episode_reward
        else:
            cumulative_rewards[episode] = cumulative_rewards[episode-1] + episode_reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_reward):.2f}")
    
    # Plot both episode rewards and cumulative rewards
    plt.figure(figsize=(12, 6))
    for i in range(n_agents):
        plt.plot(agent_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        plt.plot(cumulative_rewards[:, i]/(np.arange(n_episodes)+1), 
                label=f'Agent {i} Cumulative Avg Reward')
    
    plt.title('Agent Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('agent_rewards.png')
    plt.show()
    
    return agent_rewards, agents, env

def find_min_steps_for_all_agents(env, agents):
    """
    Function to find the minimum number of steps for all agents to reach their goal positions
    using the QLearning agents after training.
    """
    steps_to_goal = np.zeros(len(agents))  # Store steps for each agent
    
    for i, agent in enumerate(agents):
        state = env.reset()  # Reset environment and get initial positions of all agents
        agent_position = state[i]  # Get the starting position of this agent
        steps = 0
        
        while agent_position != env.goal_pos[i]:  # Loop until the agent reaches its goal
            action = agent.get_action(agent_position)
            next_state, _, _, _ = env.step([action if j == i else 0 for j in range(len(agents))])  # Get next state
            agent_position = next_state[i]  # Update agent's position
            steps += 1
        
        steps_to_goal[i] = steps  # Store the number of steps for this agent
    
    return steps_to_goal

# Run training
SEED = 42

# Run training with seed
agent_rewards, agents, env = train(seed=SEED)

steps = find_min_steps_for_all_agents(env, agents)
print(f"Minimum steps for each agent to reach their goal: {steps}")