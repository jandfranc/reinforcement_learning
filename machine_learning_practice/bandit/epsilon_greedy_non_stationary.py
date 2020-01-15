import random
import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    def __init__(self, epsilon, step_size, num_machines, actual_rewards_obj):
        self.epsilon = epsilon
        self.num_machines = num_machines
        self.machine_list = []
        self.rewards = []
        self.actual_rewards = actual_rewards_obj
        self.step_size = step_size

        for machine in range(num_machines):
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
        self.rewards[pulled_num] = self.rewards[pulled_num] - self.step_size*(self.rewards[pulled_num] - curr_reward)

        if sum(self.iteration_list) % 1000 == 0:
            random.shuffle(self.actual_rewards)
            for reward, machine in zip(self.actual_rewards, self.machine_list):
                machine.update_reward(reward)
        return curr_reward


class SlotMachine:
    def __init__(self, reward_val):
        self.reward_val = reward_val

    def pull(self):
        return np.random.normal(self.reward_val)

    def update_reward(self, reward):
        self.reward_val = reward


def run_experiment(bandit, total_iter):
    reward_list = []
    avg_reward_list = []
    for run_exp_iterator in range(0, total_iter):
        reward_list.append(bandit.action())
        avg_reward_list.append(np.mean(reward_list))
        if run_exp_iterator % 10 == 0:
            plt.plot(avg_reward_list, color='b')
            # plt.xscale('log')
            plt.grid(True, markevery=100)
            plt.ion()
            plt.show()
            plt.pause(0.1)
    return avg_reward_list


if __name__ == "__main__":
    iterations = 10000
    actual_rewards = [1, 2, 3, 4, 5]

    learner1 = BanditAgent(0.1, 1, 5, actual_rewards)
    learner2 = BanditAgent(0.1, 0.5, 5, actual_rewards)
    learner3 = BanditAgent(0.1, 1, 5, actual_rewards)

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
