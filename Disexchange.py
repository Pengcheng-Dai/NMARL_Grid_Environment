import numpy as np
import GridEnv
import GridAgent
from scipy import special
from tqdm import trange
import time
import os

# -------------------------
# -------------------------
class DecentralizedOptimizer:
    def __init__(self, grid_size, network, agent_list, state_num, action_num,
                 init_states, goal_states, gamma, horizon, epsilon=0.5,
                 consensus_alpha=0.5):
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
        self.epsilon = epsilon
        self.consensus_alpha = consensus_alpha  # 自己权重
        self.game_simulator = GridEnv.GridEnv(grid_size=self.grid_size, num_agents=self.agent_num,
                                              ini_state=self.init_states, goal=self.goal_states,
                                              adjacency=self.network)
        self.w_table = {}

        # 定义邻居列表
        self.observation_list = self.construct_observation_table(hop=self.agent_num-1)
        self.commu_list = self.construct_observation_table(hop=1)

    def construct_observation_table(self, hop):
        neighbors_list = []
        for i in range(self.agent_num):
            neighbors_i = []
            for j in range(self.agent_num):
                abs_ij = abs(i - j)
                if abs_ij <= hop or abs_ij >= self.agent_num - hop:
                    neighbors_i.append(j)
            neighbors_list.append(neighbors_i)
        return neighbors_list

    def episode(self, rate_w=0.05):
        self.game_simulator.reset()

        # rollout
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num):
                s = self.game_simulator.global_state[i]
                idx = s[0] * self.grid_size + s[1]
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

                    phi_t[c * self.state_num + s[0] * self.grid_size + s[1]] = 1.0
                    phi_tp1[c * self.state_num + s_next[0] * self.grid_size + s_next[1]] = 1.0
                    phi_t[len(self.observation_list[i]) * self.state_num + c * self.action_num + a] = 1.0
                    phi_tp1[len(self.observation_list[i]) * self.state_num + c * self.action_num + a_next] = 1.0

                r = np.sum(self.game_simulator.global_reward_history[t])
                td_error = r + self.gamma * np.dot(phi_tp1, w) - np.dot(phi_t, w)
                w += rate_w * td_error * phi_t

            self.w_table[i] = np.clip(w, -1e3, 1e3)

        # consensus step（加权平均）
        for i in range(self.agent_num):
            w_new = self.consensus_alpha * self.w_table[i].copy()
            neighbor_weight = (1 - self.consensus_alpha) / len(self.commu_list[i])
            for j in self.commu_list[i]:
                w_new += neighbor_weight * self.w_table[j]
            self.w_table[i] = w_new

    def update_params(self, rate_theta=0.01):
        for i in range(self.agent_num):
            discount = 1.0
            for t in range(self.horizon):
                obs_dim = len(self.observation_list[i]) * self.state_num \
                          + len(self.observation_list[i]) * self.action_num
                phi = np.zeros(obs_dim)

                for c, j in enumerate(self.observation_list[i]):
                    s = self.game_simulator.global_state_history[t][j]
                    a = self.game_simulator.global_action_history[t][j]
                    phi[c * self.state_num + s[0] * self.grid_size + s[1]] = 1.0
                    phi[len(self.observation_list[i]) * self.state_num + c * self.action_num + a] = 1.0

                w = self.w_table[i]
                Q_val = np.dot(phi, w)
                adv = np.clip(Q_val, -5.0, 5.0)  # 缩小裁剪

                s_i = self.game_simulator.global_state_history[t][i]
                s_idx = s_i[0] * self.grid_size + s_i[1]
                a_i = self.game_simulator.global_action_history[t][i]

                logits = self.agent_list[i].invariant_policy[s_idx]
                prob = special.softmax(logits)
                grad = -prob
                grad[a_i] += 1.0

                self.agent_list[i].invariant_policy[s_idx] += rate_theta * discount * adv * grad
                self.agent_list[i].invariant_policy = np.clip(self.agent_list[i].invariant_policy, -10, 10)

                discount *= self.gamma

    def mc_Qvalue(self, sample_num=100):
        Q_value = 0
        for _ in range(sample_num):
            self.game_simulator.reset()
            discount = 1
            for t in range(self.horizon):
                global_action = np.zeros(self.agent_num, dtype=int)
                for i in range(self.agent_num):
                    s = self.game_simulator.global_state[i]
                    idx = s[0] * self.grid_size + s[1]
                    global_action[i] = self.agent_list[i].sample_action((-1, idx))
                self.game_simulator.step(global_action)
                reward_tot = np.sum(self.game_simulator.global_reward_history[t])
                Q_value += discount * reward_tot
                discount *= self.gamma
        return Q_value / sample_num


# -------------------------
# 主函数
# -------------------------
if __name__ == '__main__':
    seed_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for t_seed in seed_list:
        np.random.seed(t_seed)

        grid_size = 5
        state_num = 25
        action_num = 5
        agent_num = 10
        horizon = 20
        gamma = 0.9
        T = 20001
        init_states = np.array([[0,0],[2,0],[4,0],[1,1],[3,1],[1,3],[3,3],[0,4],[2,4],[4,4]])
        goal_states = np.array([2,2])

        rate_w = 1e-1
        rate_theta = 5e-3

        network = np.zeros([agent_num, agent_num])
        for i in range(agent_num):
            for j in range(agent_num):
                if abs(i-j) <= 1 or abs(i-j) == agent_num-1:
                    network[i,j] = 1

        agent_list = []
        for i in range(agent_num):
            agent_list.append(GridAgent.GridAgent(state_num, action_num, horizon, random_init=True))

        Decen_optimizer = DecentralizedOptimizer(grid_size, network, agent_list,
                                                 state_num, action_num, init_states,
                                                 goal_states, gamma, horizon)

        print("开始训练时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        Decen_perform = np.zeros(T)

        for m in trange(T):
            Decen_optimizer.episode(rate_w)
            Decen_optimizer.update_params(rate_theta)
            if m % 2000 == 0:
                Decen_perform[m] = Decen_optimizer.mc_Qvalue() / agent_num
                np.save(f"./multi_agent/objective_perform_DIS_revised3_{t_seed}.npy", Decen_perform)

        print("训练结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))