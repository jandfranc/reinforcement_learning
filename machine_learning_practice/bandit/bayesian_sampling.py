import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, num_machines, actual_rewards_obj):
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = actual_rewards_obj
        self.tau = 1

        for machine in range(self.num_machines):
            self.machine_list.append(SlotMachine(self.actual_rewards[machine]))

        self.rewards = [0] * self.num_machines
        self.iteration_list = np.copy(self.rewards)
        self.lambda_ = np.copy(self.rewards)
        self.reward_sums = np.copy(self.rewards)

    def action(self):
        samples = []
        for machine_num in range(self.num_machines):
            # samples.append(np.random.normal(self.rewards[machine_num], self.variance[machine_num]))
            samples.append(np.random.randn() / np.sqrt(self.variance[machine_num]) + self.rewards[machine_num])
        pulled_num = samples.index(max(samples))
        # Get reward
        curr_reward = self.machine_list[pulled_num].pull()
        self.iteration_list[pulled_num] += 1
        self.lambda_[pulled_num] += self.tau
        self.reward_sums[pulled_num] += curr_reward
        self.rewards[pulled_num] = self.tau * self.reward_sums[pulled_num] / self.variance[pulled_num]

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
        reward_list.append(bandit.action)
        avg_reward_list.append(np.mean(reward_list))
    return avg_reward_list


if __name__ == "__main__":
    iterations = 10000
    actual_rewards = [1, 2, 3, 4, 5]

    learner1 = BanditAgent(5, actual_rewards)
    learner2 = BanditAgent(5, actual_rewards)
    learner3 = BanditAgent(5, actual_rewards)

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
    # plt.xscale('log')
    plt.show()
