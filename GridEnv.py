import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import special
import time


# 环境定义
class GridEnv:
    def __init__(self, grid_size, num_agents, ini_state, goal, adjacency, noise_max=0.1, noise_min=0.02):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.goal = np.array(goal)
        self.adjacency = adjacency
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.ini_state = ini_state
        self.reset()


    def reset(self):
        # self.agent_positions = np.array([[2, 1], [3, 1], [2, 2], [1, 0]])
        self.agent_positions = self.ini_state
        self.done = [False] * self.num_agents
        self.global_state = self.ini_state
        self.global_state_history = [self.ini_state]
        self.global_action = None
        self.global_action_history = []
        self.global_reward = None
        self.global_reward_history = []
        # self.time_counter = 0
        # return self.agent_positions.copy()

    def get_neighbors(self, agent_idx, k):

        current = set([agent_idx])
        for _ in range(k):
            next_hop = set()
            for node in current:
                neighbors = np.where(self.adjacency[node] == 1)[0]
                next_hop.update(neighbors)
            current.update(next_hop)
        return list(current)


    def step(self, actions):
        self.global_action_history.append(actions)

        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.done[i]:
                self.agent_positions[i] = self.goal
                rewards[i] = 0
                continue

            neighbors = self.get_neighbors(i, k=1)
            neighbor_at_goal = any(
                np.linalg.norm(self.agent_positions[j] - self.goal, ord=2) == 0 for j in neighbors
            )

            noise = self.noise_max - (self.noise_max - self.noise_min) * (neighbor_at_goal / max(len(neighbors), 1))

            move_map = {
                0: np.array([0, 0]),
                1: np.array([0, 1]),
                2: np.array([0, -1]),
                3: np.array([-1, 0]),
                4: np.array([1, 0])
            }
            base_move = move_map[actions[i]]

            perturbations = [
                np.array([0, 0]),
                np.array([0, 1]),
                np.array([0, -1]),
                np.array([-1, 0]),
                np.array([1, 0])
            ]
            probs = [1 - noise, noise / 4, noise / 4, noise / 4, noise / 4]
            epsilon = perturbations[np.random.choice(len(perturbations), p=probs)]

            new_pos = self.agent_positions[i] + base_move + epsilon
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.agent_positions[i] = new_pos

            if np.linalg.norm(new_pos - self.goal, ord=2) == 0:
                self.done[i] = True
                # self.agent_positions[i] = np.array([-1, -1])
                rewards[i] = 10.0
            else:
                dist = np.linalg.norm(new_pos - self.goal, ord=2)
                rewards[i] = -1-dist
        self.global_state = self.agent_positions.copy()
        self.global_reward = rewards.copy()
        self.global_state_history.append(self.global_state)
        self.global_reward_history.append(self.global_reward)
        num_unfinished = self.num_agents - sum(self.done)
        return self.global_reward, num_unfinished # output the reward