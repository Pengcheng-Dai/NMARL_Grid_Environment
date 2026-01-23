import numpy as np
import GridEnv
import math
from scipy import special






class GridAgent:
    def __init__(self, state_num, action_num, time_max, random_init=False):
        self.state_num = state_num # node_num
        self.action_num = action_num
        self.time_max = time_max

        # We store two policy tables for the time-invariant policy and the time-varying policy
        # Both use softmax parameterization
        self.invariant_policy = np.zeros((state_num, action_num))
        # print(self.invariant_policy.shape)
        # if random_init:
        #     # self.invariant_policy = np.array([[0,10], [0,10]]) # 一直往前走的策略
        #     self.invariant_policy = np.zeros([self.state_num, self.action_num]) # 调试使用
        #     # self.invariant_policy = np.random.normal(size=(self.state_num, self.action_num))
        # # self.tv_policy = np.zeros((time_max, state_num, action_num)) # time-varying plicy

    def reset(self):
        self.invariant_policy = np.zeros((state_num, action_num))
        # self.tv_policy = np.zeros((time_max, state_num, action_num))

    # sample an action under the current policy
    # expect pair = (time, state), if time-invariant, set time = -1
    # state is 0,1,,cdots,6
    def sample_action(self, pair):
        time, state = pair
        # get the softmax policy parameters
        params = self.invariant_policy[state, :]
        # print(params.shape)
        # if time != -1: # 正常这个是不用的
        #     params = self.tv_policy[time, state, :]
        # compute the probability vector
        # print("params", params)
        prob_vec = special.softmax(params)
        # print("prob_vec", prob_vec)
        # randomly select an action based on prob_vec
        action = np.random.choice(a=self.action_num, p=prob_vec) # 注意这里不减去1了！
        return action




if __name__ == '__main__':
    state_num = 2
    action_num = 2
    time_max = 10
    agent = GridAgent(state_num, action_num, time_max)
    action_list = []
    for t in range(time_max):
        action_list.append(agent.sample_action((t, 0)))
    print("action history: {}".format(action_list))

    network = np.array([[1,1,0,0,1],
               [1,1,1,0,0],
               [0,1,1,1,0],
               [0,0,1,1,1],
               [1,0,0,1,1]])

    # stationary_dist = agent.stationary_dist(network, np.array([1, 0]), 8, True)
    # print("stationary distribution: \n{}".format(stationary_dist))

