import numpy as np
import GridEnv
import math
from scipy import special
import GridAgent
from tqdm import trange
import matplotlib.pyplot as plt
import time
import os





# 分布式算法
class DecentralizedOptimizer:
    def __init__(self, grid_size, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon=0.5):
        self.grid_size = grid_size
        self.network = network
        self.agent_list = agent_list
        self.agent_num = len(agent_list)
        self.state_num = state_num
        self.action_num = action_num
        self.init_states = init_states
        self.goal_states = goal_states
        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon  # The cost for waiting one unit of time
        self.game_simulator = GridEnv.GridEnv(grid_size=self.grid_size, num_agents=self.agent_num,
                                              ini_state=self.init_states, goal=self.goal_states,
                                              adjacency=self.network)
        # hash tables to store w functions
        self.w_table = {}

        # self.averaged_Q_table = {}
        # self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        # 定义每个智能体的观测邻居
        self.observation_list = self.construct_obervation_table(hop=agent_num-1) # 全局状态
        self.action_list = self.construct_obervation_table(hop=agent_num-1) # 全局动作
        self.reward_list = self.construct_obervation_table(hop=0)
        self.commu_list = self.construct_obervation_table(hop=1) # 策略参数交互网络

    # find neighbors of agent
    # 环形网络
    def construct_obervation_table(self, hop):
        neighbors_list = []
        for i in range(self.agent_num):
            neighbors_i = []
            for j in range(self.agent_num):
                abs_ij = abs(i - j)
                if abs_ij <= hop or abs_ij >= self.agent_num - hop:
                    neighbors_i.append(j)
            neighbors_list.append(neighbors_i)
        return neighbors_list  # [[], [], [], []...]

    # simulate the trajectory for one epsiode, and update the local Q functions
    # rate_w and rate_zeta are the learning rate of weights and eligibility vectors
    # rate_zeta is TD(0)
    def episode(self, rate_w):
        self.game_simulator.reset()

        # rollout
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num):
                s = self.game_simulator.global_state[i]
                idx = s[0] * 5 + s[1]
                global_action[i] = self.agent_list[i].sample_action((-1, idx))
            self.game_simulator.step(global_action)

        # TD(0) critic update
        for i in range(self.agent_num):
            obs_dim = len(self.observation_list[i]) * self.state_num \
                      + len(self.observation_list[i]) * self.action_num
            w = self.w_table.get(i, np.zeros(obs_dim))

            for t in range(self.horizon - 1):
                phi_t = np.zeros(obs_dim)
                phi_tp1 = np.zeros(obs_dim)

                for c, j in enumerate(self.observation_list[i]):
                    s = self.game_simulator.global_state_history[t][j]
                    s_next = self.game_simulator.global_state_history[t + 1][j]

                    a = self.game_simulator.global_action_history[t][j]
                    a_next = self.game_simulator.global_action_history[t + 1][j]

                    phi_t[c * self.state_num + s[0] * 5 + s[1]] = 1.0
                    phi_tp1[c * self.state_num + s_next[0] * 5 + s_next[1]] = 1.0

                    phi_t[len(self.observation_list[i]) * self.state_num
                          + c * self.action_num + a] = 1.0
                    phi_tp1[len(self.observation_list[i]) * self.state_num
                            + c * self.action_num + a_next] = 1.0

                # centralized reward（先保证信号）
                r = np.sum(self.game_simulator.global_reward_history[t])

                td_error = r + self.gamma * np.dot(phi_tp1, w) - np.dot(phi_t, w)
                w += rate_w * td_error * phi_t

            self.w_table[i] = np.clip(w, -1e3, 1e3)
        # consensus step
        for i in range(self.agent_num):
            w_new = (1/3) * self.w_table[i].copy()
            for j in self.commu_list[i]:
                w_new += (1/3) * self.w_table[j]
            self.w_table[i] = w_new

    # 更新策略参数
    def update_params(self, rate_theta):
        for i in range(self.agent_num):
            discount = 1.0

            for t in range(self.horizon):
                obs_dim = len(self.observation_list[i]) * self.state_num \
                          + len(self.observation_list[i]) * self.action_num
                phi = np.zeros(obs_dim)

                for c, j in enumerate(self.observation_list[i]):
                    s = self.game_simulator.global_state_history[t][j]
                    a = self.game_simulator.global_action_history[t][j]

                    phi[c * self.state_num + s[0] * 5 + s[1]] = 1.0
                    phi[len(self.observation_list[i]) * self.state_num
                        + c * self.action_num + a] = 1.0

                w = self.w_table[i]
                Q_val = np.dot(phi, w)

                # advantage clipping
                adv = np.clip(Q_val, -10.0, 10.0)

                s_i = self.game_simulator.global_state_history[t][i]
                s_idx = s_i[0] * 5 + s_i[1]
                a_i = self.game_simulator.global_action_history[t][i]

                logits = self.agent_list[i].invariant_policy[s_idx]
                prob = special.softmax(logits)

                grad = -prob
                grad[a_i] += 1.0

                self.agent_list[i].invariant_policy[s_idx] += \
                    rate_theta * discount * adv * grad

                self.agent_list[i].invariant_policy = np.clip(
                    self.agent_list[i].invariant_policy, -10, 10
                )

                discount *= self.gamma

    # 估计Q函数的值
    # 300个episode取均值
    def mc_Qvalue(self, sample_num=300):
        Q_value = 0
        for epis in range(sample_num):
            self.game_simulator.reset()
            discount = 1
            # run an episode and record the trajectory
            for t in range(self.horizon):  # 一个episode
                global_action = np.zeros(self.agent_num, dtype=int)
                for i in range(self.agent_num):  # 智能体选取动作
                    global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i,0]* 5 + self.game_simulator.global_state[i,1]))
                self.game_simulator.step(global_action)
                reward_tot = 0
                for i in range(self.agent_num):
                    reward_tot += self.game_simulator.global_reward_history[t][i]
                Q_value += discount * reward_tot
                discount = discount * self.gamma
        Q_value = Q_value / sample_num
        return Q_value



if __name__ == '__main__':
    # seed_list = [0]
    seed_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for t_seed in seed_list:
    np.random.seed(t_seed)
    #     t_seed = 0
    #     np.random.seed(t_seed)
    grid_size = 5
    state_num = 25
    action_num = 5
    agent_num = 10  # 智能体的个数
    horizon = 20  # SAC中的样本取样都是10
    gamma = 0.9
    T = 60001
    init_states = np.array([[0, 0], [2, 0], [4, 0], [1, 1], [3, 1], [1, 3], [3, 3], [0, 4], [2, 4], [4, 4]])
    goal_states = np.array([2, 2])

    rate_w = 5e-2
    rate_theta = 5e-4

    network = np.zeros([agent_num, agent_num])
    for i in range(agent_num):
        for j in range(agent_num):
            if abs(i - j) <= 1 or abs(i - j) == agent_num - 1:
                network[i, j] = 1

    agent_list = []
    for i in range(agent_num):
        agent_list.append(GridAgent.GridAgent(state_num, action_num, horizon, random_init=True))

    #####################################################################################
    # # # centralized algorithm
    Decen_optimizer = DecentralizedOptimizer(grid_size, network, agent_list, state_num, action_num, init_states, goal_states,
                                   gamma, horizon)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("开始时间：", formatted_time)

    Decen_perform = np.zeros(T)
    # SAC_aver_gradient = np.zeros(T)
    for m in trange(T):
        # print("m", m)s
        Decen_optimizer.episode(rate_w)
        # update of the parameters
        Decen_optimizer.update_params(rate_theta)
        # 每个400步评估一次
        if m % 4000 == 0:
            Decen_perform[m] = Decen_optimizer.mc_Qvalue() / agent_num
        # 保存关键数据
        np.save("./multi_agent/objective_perform_DIS_{}.npy".format(t_seed), Decen_perform)
        # np.save("./multi_agent_14_2/aver_gradient_EDR_maxmore{}.npy".format(t_seed), EDR_aver_gradient)

    # 输出结束时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("结束时间：", formatted_time)
