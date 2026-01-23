import numpy as np
import GridEnv
import math
from scipy import special
import GridAgent
from tqdm import trange
import matplotlib.pyplot as plt
import time
import os





# give a global state in nd_array
# compute the global state code

# 把全局状态向量用数字进行编码
def global_state_encoder(global_state, state_num):
    agent_num = global_state.shape[0]
    global_state_code = 0
    for i in range(agent_num):
        global_state_code *= state_num
        global_state_code += (global_state[i,0] * 5 + global_state[i,1])
    return global_state_code


def global_action_encoder(global_action, action_num):
    agent_num = global_action.shape[0]
    global_action_code = 0
    for i in range(agent_num):
        global_action_code *= action_num
        global_action_code += global_action[i]
    return global_action_code



# 对比算法--Scalable Actor Critic
class SACOptimizer:
    def __init__(self, grid_size, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon,
                 epsilon=0.5):
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

        self.local_Q_table = {}  # Q_{i}(s_{\mathcal{N}^{\kappa}_{i}},a_{\mathcal{N}^{\kappa}_{i}})
        # find the neighbors of agent
        self.observation_list = self.construct_obervation_table(hop=0)
        self.action_list = self.construct_obervation_table(hop=0)


    # find neighbors of agent
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
    def local_episode(self, rate_w):
        self.game_simulator.reset()
        TD_y = np.zeros(self.horizon - 1)
        # run an episode and record the trajectory
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num):  # choice the action
                global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i,0]* 5 + self.game_simulator.global_state[i,1]))
            self.game_simulator.step(global_action)
        # update the local Q functions
        for t in range(self.horizon - 1):
            # compute Q_{tot}(s,a) first

            for i in range(self.agent_num):
                local_Q_value = self.local_Q_table.get(i, np.zeros([self.state_num ** len(self.observation_list[i]),
                                                                    self.action_num ** len(self.observation_list[i])]))
                # print(local_Q_value.shape)
                local_state_t = []  # s_{\mathcal{N}_{i},t}
                local_action_t = []  # a_{\mathcal{N}_{i},t}
                local_state_t_1 = []  # s_{\mathcal{N}_{i},t+1}
                local_action_t_1 = []  # a_{\mathcal{N}_{i},t+1}
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]  # 第i个智能体的邻居j
                    # 邻居j的状态
                    local_state_t.append(self.game_simulator.global_state_history[t][j])
                    local_action_t.append(self.game_simulator.global_action_history[t][j])
                    local_state_t_1.append(self.game_simulator.global_state_history[t + 1][j])
                    local_action_t_1.append(self.game_simulator.global_action_history[t + 1][j])
                # print(local_action_t)
                # print(local_action_t_1)
                # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                local_action_t_code = global_action_encoder(np.array(local_action_t), self.action_num) # DAi
                local_state_t_1_code = global_state_encoder(np.array(local_state_t_1), self.state_num)
                local_action_t_1_code = global_action_encoder(np.array(local_action_t_1), self.action_num) # Dai
                TD_y[t] += self.game_simulator.global_reward_history[t][i] + self.gamma * local_Q_value[local_state_t_1_code, local_action_t_1_code]

            # update Q table
            for i in range(self.agent_num):
                local_Q_value = self.local_Q_table.get(i, np.zeros([self.state_num ** len(self.observation_list[i]),
                                                                        self.action_num ** len(
                                                                            self.observation_list[i])]))
                local_Q_value[local_state_t_code, local_action_t_code] = (1 - rate_w) * local_Q_value[
                    local_state_t_code, local_action_t_code] + rate_w * (TD_y[t]/self.agent_num)
                self.local_Q_table[i] = local_Q_value

    # 更新策略参数
    def update_params(self, rate_theta):
        for i in range(self.agent_num):
            discount_factor = 1.0
            for t in range(self.horizon):
                # first compute the Q function value
                # 根据样本的history取样
                local_state_t = []
                local_action_t = []
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]  # 第i个智能体的邻居j
                    # 邻居j的状态
                    local_state_t.append(self.game_simulator.global_state_history[t][j])
                    local_action_t.append(self.game_simulator.global_action_history[t][j])
                # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                local_action_t_code = global_action_encoder(np.array(local_action_t), self.action_num) # Dai
                # 取对应的Q值
                localQi = self.local_Q_table.get(i, np.zeros([self.state_num ** len(self.observation_list[i]),
                                                              self.action_num ** len(self.observation_list[i])]))
                localQivalue = localQi[local_state_t_code, local_action_t_code]
                # 邻居智能体的局部Q值
                tot_localQvalue = [localQivalue]
                for j in self.observation_list[i]:
                    if j is not i:  # 邻居中不包含i
                        local_state_t = []
                        local_action_t = []
                        for c in range(len(self.observation_list[j])):
                            k = self.observation_list[j][c]  # 第j个智能体的邻居k
                            # 邻居j的状态
                            local_state_t.append(self.game_simulator.global_state_history[t][k])
                            local_action_t.append(self.game_simulator.global_action_history[t][k])
                        # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                        local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                        local_action_t_code = global_action_encoder(np.array(local_action_t), self.action_num) # Dai
                        # 取对应的Q值
                        localQj = self.local_Q_table.get(j, np.zeros([self.state_num ** len(self.observation_list[j]),
                                                                      self.action_num ** len(
                                                                          self.observation_list[j])]))
                        localQjvalue = localQj[local_state_t_code, local_action_t_code]
                        tot_localQvalue.append(localQjvalue)
                # 开始计算梯度，更新策略参数
                local_state = self.game_simulator.global_state_history[t][i]
                local_action = self.game_simulator.global_action_history[t][i]
                params = self.agent_list[i].invariant_policy[local_state[0] * 5 + local_state[1], :]
                prob_vec = special.softmax(params)
                term1 = np.zeros(self.action_num)
                # term1[local_action + 1] = 1.0
                term1[local_action] = 1.0
                term1 -= prob_vec
                self.agent_list[i].invariant_policy[local_state[0] * 5 + local_state[1], :] += (
                        rate_theta * discount_factor * (np.sum(tot_localQvalue) / self.agent_num) * term1)
                discount_factor *= self.gamma


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
        Q_value = Q_value/sample_num
        return Q_value





if __name__ == '__main__':
    # seed_list = [0]
    seed_list = [0,10,20,30,40,50,60,70,80,90]
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

    rate_w = 1e-2
    rate_theta = 1e-2

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
    SAC_optimizer = SACOptimizer(grid_size, network, agent_list, state_num, action_num, init_states, goal_states,
                                 gamma, horizon)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("开始时间：", formatted_time)


    SAC_objective_perform = np.zeros(T)
    SAC_aver_gradient = np.zeros(T)
    for m in trange(T):
        # print("m", m)
        SAC_optimizer.local_episode(rate_w)
        # update of the parameters
        SAC_optimizer.update_params(rate_theta)
        # 每个400步评估一次
        if m % 4000 == 0:
            SAC_objective_perform[m] = SAC_optimizer.mc_Qvalue() / agent_num
        # 保存关键数据
        np.save("./multi_agent/objective_perform_VDN_IP_more{}.npy".format(t_seed), SAC_objective_perform)
        # np.save("./multi_agent_14_2/aver_gradient_EDR_maxmore{}.npy".format(t_seed), EDR_aver_gradient)

    # 输出结束时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("结束时间：", formatted_time)
