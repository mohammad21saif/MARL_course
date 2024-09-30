import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple

class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

        # Initialize attributes
        self.steps: int = 0
        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        self.steps: int = 0
        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = self.distances[self.loc].tolist()

        state = np.concatenate(
            (
                np.array([self.loc]),
                np.array(self.dist),
                np.array(self.locations).reshape(-1),
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc].tolist()
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                np.array(next_dist),
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> np.ndarray:
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: List) -> np.ndarray:
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        if next_loc not in self.visited_targets:
            reward = -self.distances[past_loc][next_loc]
        else:
            reward = -10000
        return reward


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    V = np.zeros(num_states)
    for i in range(max_iterations):
        delta = 0
        for s in range(num_states):
            v = V[s]
            Q = np.zeros(num_actions)
            for a in range(num_actions):
                next_state, reward, terminated, truncated, _ = env.step(a)
                Q[a] = reward + gamma * V[int(next_state[0])]  # Use the location as the state index
                env.reset()  # Reset the environment after each action
            V[s] = np.max(Q)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q = np.zeros(num_actions)
        for a in range(num_actions):
            next_state, reward, terminated, truncated, _ = env.step(a)
            Q[a] = reward + gamma * V[int(next_state[0])]
            env.reset()
        policy[s] = np.argmax(Q)
    
    return V, policy



if __name__ == "__main__":
    num_targets = 50 
    env = TSP(num_targets)
    
    print("Starting Value Iteration...")
    optimal_value, optimal_policy = value_iteration(env)
    
    print("Evaluating the policy...")
    total_reward = 0
    state, _ = env.reset()
    for _ in range(num_targets):
        action = optimal_policy[int(state[0])]
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"Total reward using Value Iteration: {total_reward}")
    print(f"Optimal policy: {optimal_policy}")