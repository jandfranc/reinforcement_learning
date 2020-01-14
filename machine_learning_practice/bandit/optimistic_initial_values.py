import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, initial_value, num_machines, actual_rewards_obj):
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = actual_rewards_obj
        self.initial_val = initial_value

        for machine in range(num_machines):
            self.machine_list.append(SlotMachine(self.actual_rewards[machine]))
            self.rewards.append(self.initial_val)
        self.iteration_list = np.copy(self.rewards)

    def action(self):
        pulled_num = self.rewards.index(max(self.rewards))
        # Get reward
        curr_reward = self.machine_list[pulled_num].pull()
        # Update Mean
        self.iteration_list[pulled_num] += 1
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

    learner1 = BanditAgent(20, 5, actual_rewards)
    learner2 = BanditAgent(10, 5, actual_rewards)
    learner3 = BanditAgent(0, 5, actual_rewards)

    avg_rewards1 = run_experiment(learner1, iterations)
    avg_rewards2 = run_experiment(learner2, iterations)
    avg_rewards3 = run_experiment(learner3, iterations)

    print(learner1.actual_rewards)
    print(learner1.rewards)
    print(learner2.actual_rewards)
    print(learner2.rewards)
    print(learner3.actual_rewards)
    print(learner3.rewards)

    plt.plot(avg_rewards1, color='b')
    plt.plot(avg_rewards2, color='g')
    plt.plot(avg_rewards3, color='r')
    plt.xscale('log')
    plt.show()
