from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the ModTSP environment as provided
class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets to maximize profits. The profits decay with time.
    """

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): No. of targets the agent needs to visit.
            max_area (int): Max. Square area where the targets are defined.
            shuffle_time (int): No. of episodes after which the profits are to be shuffled.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.steps: int = 0
        self.episodes: int = 0

        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: npt.NDArray[np.float32] = self._generate_points(self.num_targets)
        self.distances: npt.NDArray[np.float32] = self._calculate_distances(self.locations)

        # Initialize profits for each target
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        # Simplify the state representation to make it manageable for function approximation
        # We'll include the current location, visited targets, and current profits
        self.state_dim = 1 + self.num_targets + self.num_targets
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # Action Space : {next_target}
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Seed to reset the environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and an empty info dictionary.
        """
        self.steps: int = 0
        self.episodes += 1

        self.loc: int = 0
        self.visited_targets: npt.NDArray[np.float32] = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()
        self.dist: List = self.distances[self.loc]

        if self.shuffle_time != 0 and self.episodes % self.shuffle_time == 0:
            np.random.shuffle(self.initial_profits)

        # Simplify the state representation
        state = np.concatenate(
            (
                np.array([self.loc / (self.num_targets - 1)]),  # Normalize current location index
                self.visited_targets,
                self.current_profits / np.max(self.initial_profits),  # Normalize profits
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target).

        Args:
            action (int): The index of the next target to move to.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
                - The new state of the environment.
                - The reward for the action.
                - A boolean indicating whether the episode has terminated.
                - A boolean indicating if the episode is truncated.
                - An empty info dictionary.
        """
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        self.current_profits -= self.distances[past_loc, next_loc]
        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        # Simplify the state representation
        next_state = np.concatenate(
            [
                np.array([next_loc / (self.num_targets - 1)]),
                self.visited_targets,
                self.current_profits / np.max(self.initial_profits),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> npt.NDArray[np.float32]:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate the distance matrix between all target locations.

        Args:
            locations: List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


# Define the Q-Learning Agent with function approximation
class QLearningAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Linear approximator for Q-values
        self.q_network = nn.Linear(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # Compute current Q value
        q_value = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute next Q value
        with torch.no_grad():
            next_q_value = self.q_network(next_state).max(1)[0]

        # Compute target Q value
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Compute loss
        loss = self.loss_fn(q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# # Define the plotting function
# def plot_episode(locations, path, episode, rewards):
#     """Plot the episode's path and target locations.

#     Args:
#         locations (ndarray): Array of target locations.
#         path (list): Sequence of target indices visited by the agent.
#         episode (int): The current episode number.
#         rewards (float): Total rewards obtained in the episode.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.scatter(locations[:, 0], locations[:, 1], color='blue', label='Targets')

#     if len(path) > 1:
#         # Plot the visited path
#         path_locs = locations[path]
#         plt.plot(path_locs[:, 0], path_locs[:, 1], color='green', linestyle='-', marker='o', label='Agent Path')

#     for i, loc in enumerate(locations):
#         plt.text(loc[0] + 0.1, loc[1] + 0.1, str(i), color='red', fontsize=12)  # Label targets with index

#     plt.title(f"Episode {episode}: Total Reward = {rewards}")
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# Define the main function with Q-Learning
def main() -> None:
    """Main function to train the Q-Learning agent."""
    num_targets = 10
    env = ModTSP(num_targets)
    state, _ = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters
    num_episodes = 10000
    lr = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 300
    plot_every = 50  # Plot every 50 episodes

    # Initialize Agent
    agent = QLearningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    ep_rets = []

    for ep in range(1, num_episodes + 1):
        ret = 0
        state, _ = env.reset()
        path = []  # To keep track of the path for plotting

        for _ in range(env.max_steps):
            action = agent.select_action(state)
            path.append(int(state[0] * (env.num_targets - 1)))  # Track the current position

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward

            # Update agent
            agent.update(state, action, reward, next_state, float(done))

            # Move to the next state
            state = next_state

            if done:
                break

        ep_rets.append(ret)

        # Print progress
        if ep % 10 == 0:
            print(f"Episode {ep} : Total Reward = {ret:.2f} | Epsilon = {agent.epsilon:.3f}")

        # # Plot the agent's path periodically
        # if ep % plot_every == 0:
        #     plot_episode(env.locations, path, ep, ret)

    # Final evaluation
    print(f"Average return over {num_episodes} episodes: {np.mean(ep_rets):.2f}")

    # Plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(ep_rets, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning on ModTSP: Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
