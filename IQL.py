import numpy as np
import GridEnv
import math
from scipy import special
import GridAgent
from tqdm import trange
import matplotlib.pyplot as plt
import time
import os





class Agent:
    def __init__(self, state_num, action_num, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((state_num, action_num)) # 注意一下
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_num = state_num
        self.action_num = action_num

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_num)
        return np.argmax(self.Q[state[0] * 5 + state[1]])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state[0] * 5 + next_state[1]])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state[0] * 5 + state[1], action]
        self.Q[state[0] * 5 + state[1], action] += self.alpha * td_error






# 300个episode取均值
def mc_Qvalue(game, agent_num, agents, gamma, sample_num=300):
    Q_value = 0
    for epis in range(sample_num):
        game.reset()
        discount = 1
        # run an episode and record the trajectory
        for t in range(20):  # 一个episode
            state = game.global_state.copy()
            global_action = np.array([np.argmax(agent.Q[state[i, 0] * 5 + state[i, 1]]) for i, agent in enumerate(agents)])
            #     global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i]))
            game.step(global_action)
            reward_tot = 0
            for i in range(agent_num):
                reward_tot += game.global_reward_history[t][i]
            Q_value += discount * reward_tot
            discount = discount * gamma
    Q_value = Q_value/sample_num
    return Q_value




# def mtest_trained_agents(game, agents, max_steps=20):
#     game.reset()
#     print("Initial State:", game.global_state)
#     for step in range(max_steps):
#         state = game.global_state.copy()
#         actions = np.array([np.argmax(agent.Q[state[i]]) for i, agent in enumerate(agents)])
#         rewards, unfinished = game.step(actions)
#         print(f"Step {step+1}: State = {game.global_state}, Reward = {rewards}, Unfinished = {unfinished}")
#         if unfinished == 0:
#             print("All agents reached goal!")
#             break


if __name__ == '__main__':
    seed_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for t_seed in seed_list:
    np.random.seed(t_seed)
    #     t_seed = 0
    grid_size = 5
    state_num = 25
    action_num = 5
    agent_num = 10  # 智能体的个数
    horizon = 20  # SAC中的样本取样都是10
    gamma = 0.9
    T = 60001
    init_states = np.array([[0, 0], [2, 0], [4, 0], [1, 1], [3, 1], [1, 3], [3, 3], [0, 4], [2, 4], [4, 4]])
    goal_states = np.array([2, 2])

    rate_w = 1e-2
    rate_theta = 1e-2

    network = np.zeros((agent_num, agent_num))
    for i in range(agent_num):
        for j in range(agent_num):
            if abs(i - j) <= 1 or abs(i - j) == agent_num - 1:
                network[i, j] = 1

    game = GridEnv.GridEnv(grid_size=grid_size, num_agents=agent_num,
                                              ini_state=init_states, goal=goal_states,
                                              adjacency=network)

    # print("Training...")
    num_episodes = 60001
    max_steps = 20
    agents = [Agent(state_num, action_num) for _ in range(agent_num)]
    IQL_objective_perform = np.zeros(num_episodes)
    for episode in trange(num_episodes):
        if episode % 4000 == 0:
            IQL_objective_perform[episode] = mc_Qvalue(game, agent_num, agents, gamma, sample_num=300)
        game.reset()
        for step in range(max_steps):
            current_states = game.global_state.copy()
            actions = np.array([agents[i].select_action(current_states[i]) for i in range(agent_num)])
            rewards, unfinished = game.step(actions)
            next_states = game.global_state.copy()

            for i in range(agent_num):
                agents[i].update(current_states[i], actions[i], rewards[i], next_states[i])

            if unfinished == 0:
                break
            # 保存关键数据
            np.save("./multi_agent/objective_perform_IQL{}.npy".format(t_seed), IQL_objective_perform)
    # print("Testing Trained Agents...")
    # print("result", Q_value)