import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log


class BanditAgent:
    def __init__(self, num_machines, actual_rewards_obj):
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = actual_rewards_obj
        self.total_iterations = 1

        for machine in range(num_machines):
            self.machine_list.append(SlotMachine(self.actual_rewards[machine]))
            self.rewards.append(0)
        self.iteration_list = np.copy(self.rewards)

    def action(self):
        ucb_x = [0] * self.num_machines
        for reward_num, reward in enumerate(self.rewards):
            ucb_x[reward_num] = reward + sqrt(2 * log(self.total_iterations) / (self.iteration_list[reward_num] + 1))
        pulled_num = ucb_x.index(max(ucb_x))
        # Get reward
        curr_reward = self.machine_list[pulled_num].pull()
        # Update Mean
        self.iteration_list[pulled_num] += 1
        self.total_iterations += 1
        one_over_iter = 1 / (self.iteration_list[pulled_num])
        self.rewards[pulled_num] = (1 - one_over_iter) * self.rewards[pulled_num] + one_over_iter * curr_reward

        return curr_reward


class SlotMachine:
    def __init__(self, reward_val):
        self.reward_val = reward_val

    def pull(self):
        return np.random.normal(self.reward_val)


def run_experiment(bandit, total_iter):
    reward_list = []
    avg_reward_list = []
    for run_exp_iterator in range(0, total_iter):
        reward_list.append(bandit.action())
        avg_reward_list.append(np.mean(reward_list))
    return avg_reward_list


if __name__ == "__main__":
    iterations = 10000
    actual_rewards = [1, 2, 3, 4, 5]

    learner1 = BanditAgent(5, actual_rewards)

    avg_rewards1 = run_experiment(learner1, iterations)

    print(learner1.actual_rewards)
    print(learner1.rewards)
    plt.plot(avg_rewards1, color='b')
    # plt.xscale('log')
    plt.show()
