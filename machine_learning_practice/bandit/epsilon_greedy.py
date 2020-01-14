import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, epsilon, num_machines, actual_rewards):
        self.epsilon = epsilon
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = actual_rewards

        for machine in range(num_machines):
            self.actual_rewards.append(self.actual_rewards[machine])
            self.machine_list.append(SlotMachine(self.actual_rewards[machine]))
            self.rewards.append(0)
        self.iteration_list = np.copy(self.rewards)

    def action(self):
        rand_val = np.random.uniform(0.0, 1.0)
        # Explore
        if rand_val < self.epsilon:
            pulled_num = np.random.randint(self.num_machines)
        # Exploit
        else:
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


def run_experiment(bandit, iterations):
    reward_list = []
    avg_reward_list = []
    for run_exp_iterator in range(0, iterations):
        reward_list.append(bandit.action())
        avg_reward_list.append(np.mean(reward_list))
    return avg_reward_list


if __name__ == "__main__":
    iterations = 10000
    actual_rewards = [1, 2, 3, 4, 5]
    learner1 = BanditAgent(0.1, 5, actual_rewards)
    learner2 = BanditAgent(0.01, 5, actual_rewards)
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
    \plt.xscale('log')
    plt.show()
